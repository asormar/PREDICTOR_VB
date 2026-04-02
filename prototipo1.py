import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 0. NORMALIZACIÓN DE NOMBRES DE CLUBES
# ─────────────────────────────────────────────

CLUB_NAME_MAP = {
    # Flashscore/Archivos -> Tu CSV de comparación (10 años)
    "Lube Civitanova": "Lube",
    "LubeCivitanova": "Lube",
    "Trentino": "Trento",
    "Sir Safety Perugia": "Perugia",
    "Vero Volley Monza": "Monza",
    "Allianz Milano": "Milano",
    "Powervolley Milano": "Milano",
    "Piacenza Copra": "Piacenza",
    "Gas Sales Piacenza": "Piacenza",
    "Cuneo Volley": "Cuneo",
    "Vibo Valentia": "Vibo Valentia",
    "Castellana Grotte": "Castellana Grotte New Mater",
    "Acicastello": "Acicastello",
    "Saturnia Acicastello": "Acicastello"
}

def normalize_club(name: str) -> str:
    if not isinstance(name, str): return name
    # Limpiar nombres duplicados (Lube CivitanovaLube Civitanova -> Lube Civitanova)
    # Buscamos si la cadena contiene una repetición exacta
    s = name.strip()
    mid = len(s) // 2
    if len(s) > 4 and s[:mid] == s[mid:]:
        s = s[:mid]
    
    # Aplicar mapa de nombres
    return CLUB_NAME_MAP.get(s, s)


# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────

def load_team_data(path: str) -> pd.DataFrame:
    """Carga el archivo de comparación de equipos (Comparacion_equipos_10_años)."""
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Club_Club":                      "club",
        "Played Matches_Played Matches":  "matches",
        "Played Set_Played Set":          "sets_played",
        "POINTS_Tot":                     "points_tot",
        "POINTS_BP":                      "points_bp",
        "SERVE_Tot":                      "serve_tot",
        "SERVE_Ace":                      "serve_ace",
        "SERVE_Err.":                     "serve_err",
        "SERVE_Ace per Set":              "serve_ace_per_set",
        "SERVE_Effic.":                   "serve_effic",
        "RECEPTION_Tot":                  "rec_tot",
        "RECEPTION_Err.":                 "rec_err",
        "RECEPTION_Neg.":                 "rec_neg",
        "RECEPTION_Exc.":                 "rec_exc",
        "RECEPTION_Exc. %":               "rec_exc_pct",
        "RECEPTION_Effic.":               "rec_effic",
        "ATTACK_Tot":                     "att_tot",
        "ATTACK_Err.":                    "att_err",
        "ATTACK_Blocked":                 "att_blocked",
        "ATTACK_Exc.":                    "att_exc",
        "ATTACK_Exc. %":                  "att_exc_pct",
        "ATTACK_Effic.":                  "att_effic",
        "BLOCK_Exc.":                     "block_exc",
        "BLOCK_Points per Set":           "block_per_set",
        "Temporada":                      "season",
    })

    # Normalización mejorada de temporada
    def fix_season(x):
        if isinstance(x, (int, float, np.integer)):
            # Si el Excel dice 2025, lo convertimos a "2025/2026" para que coincida con Flashscore
            return f"{int(x)}/{int(x)+1}"
        return str(x)

    df["season"] = df["season"].apply(fix_season)
    df["club"] = df["club"].apply(normalize_club)
    return df


# 2. Carga de jugadores corregida (Extrae el nombre del archivo)
def load_player_data(paths: list) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            # ¡IMPORTANTE! Extraemos el nombre del equipo del nombre del archivo
            # Ej: "LubeCivitanova_historial_10_años.csv" -> "Lube Civitanova"
            filename = os.path.basename(p)
            team_name_from_file = filename.split('_')[0]
            
            df['ID_Equipo'] = team_name_from_file
            frames.append(df)
        except Exception as e:
            print(f"Error cargando {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_matches(paths: list) -> pd.DataFrame:
    """
    Carga y concatena varios archivos de enfrentamientos directos.
    Normaliza nombres de clubes automáticamente.
    Espera columnas: season, home_club, away_club, home_sets, away_sets
    """
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["home_club"] = df["home_club"].apply(normalize_club)
    df["away_club"] = df["away_club"].apply(normalize_club)
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING DE EQUIPOS
# ─────────────────────────────────────────────

def build_team_features(df_teams: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features por (club, season).
    Añade ratios por set y métricas derivadas.
    """
    df = df_teams.copy()

    sp = df["sets_played"].replace(0, np.nan)

    # Ratios por set (normaliza volumen de partidos)
    df["serve_ace_rate"] = df["serve_ace"]   / sp
    df["att_err_rate"]   = df["att_err"]     / sp
    df["block_rate"]     = df["block_exc"]   / sp
    df["rec_err_rate"]   = df["rec_err"]     / sp
    df["att_blocked_rate"] = df["att_blocked"] / sp

    # Eficiencia combinada: proxy de presión ofensiva
    df["attack_pressure"] = df["att_effic"] - df["rec_err_rate"]

    # Ratio puntos de rotura sobre total (mide constancia en momentos clave)
    df["bp_ratio"] = df["points_bp"] / df["points_tot"].replace(0, np.nan)

    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING DE JUGADORES
# ─────────────────────────────────────────────

# 3. Función de construcción de features de jugadores (Sincronizada)
def build_player_features(df_players: pd.DataFrame) -> pd.DataFrame:
    if df_players.empty: return pd.DataFrame()
    
    df = df_players.copy()
    df.columns = df.columns.str.strip()
    
    # Mapeo de columnas
    df = df.rename(columns={
        "Played Set_Played Set": "sets_played",
        "ID_Equipo": "club",
        "Temporada": "season",
        "ATTACK_Effic.": "att_effic",
        "SERVE_Ace per Set": "serve_ace_per_set",
        "RECEPTION_Effic.": "rec_effic",
        "BLOCK_Points per Set": "block_per_set"
    })

    # Normalizar temporada: 2025/2026 -> 2025/2026 (se queda igual si ya tiene el formato)
    # Pero si viene como 2025, lo cambiamos:
    def fix_season_player(x):
        x = str(x)
        if '/' not in x:
            return f"{x}/{int(x)+1}"
        return x
    
    df["season"] = df["season"].apply(fix_season_player)
    df["club"] = df["club"].apply(normalize_club)
    df["weight"] = pd.to_numeric(df["sets_played"], errors='coerce').fillna(0)

    PLAYER_COLS = ["att_effic", "serve_ace_per_set", "rec_effic", "block_per_set"]

    records = []
    for (club, season), g in df.groupby(["club", "season"]):
        rec = {"club": club, "season": season}
        for col in PLAYER_COLS:
            if col in g.columns:
                w = g["weight"]
                v = pd.to_numeric(g[col], errors='coerce').fillna(0)
                rec[f"player_{col}_wmean"] = (v * w).sum() / w.sum() if w.sum() > 0 else 0
        records.append(rec)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 4. CONSTRUCCIÓN DEL DATASET DE PARTIDOS
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # Equipo
    "serve_ace_rate", "serve_effic",
    "rec_exc_pct", "rec_effic", "rec_err_rate",
    "att_exc_pct", "att_effic", "att_err_rate", "att_blocked_rate",
    "block_rate", "block_per_set",
    "attack_pressure", "bp_ratio",
    # Jugadores (agregados)
    "player_att_effic_wmean", "player_serve_ace_per_set_wmean",
    "player_rec_effic_wmean", "player_block_per_set_wmean",
    "player_serve_effic_wmean", "player_att_exc_pct_wmean",
    "player_rec_exc_pct_wmean",
]


def get_team_vector(team_feat: pd.DataFrame, club: str, season: str) -> pd.Series:
    """Extrae el vector de features de un equipo en una temporada."""
    row = team_feat[(team_feat["club"] == club) & (team_feat["season"] == season)]
    if row.empty:
        return pd.Series({c: np.nan for c in FEATURE_COLS})
    # Solo devuelve las columnas que existen
    available = [c for c in FEATURE_COLS if c in row.columns]
    result = pd.Series({c: np.nan for c in FEATURE_COLS})
    result.update(row[available].iloc[0])
    return result


def build_match_dataset(
    df_teams: pd.DataFrame,
    df_player_features: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Para cada partido genera un vector diferencial (local − visitante)
    y los targets home_sets / away_sets.
    """
    team_feat = build_team_features(df_teams)

    if not df_player_features.empty:
        team_feat = team_feat.merge(df_player_features, on=["club", "season"], how="left")

    records = []
    for _, match in matches_df.iterrows():
        home_f = get_team_vector(team_feat, match["home_club"], match["season"])
        away_f = get_team_vector(team_feat, match["away_club"], match["season"])
        diff   = home_f - away_f

        record = diff.add_prefix("diff_").to_dict()
        record["season"]    = match["season"]
        record["home_club"] = match["home_club"]
        record["away_club"] = match["away_club"]
        record["home_sets"] = match["home_sets"]
        record["away_sets"] = match["away_sets"]
        records.append(record)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 5. MODELO
# ─────────────────────────────────────────────

class SetPredictor:
    """
    Predice sets ganados por el equipo local y visitante.
    Entrena dos modelos LightGBM independientes.
    """

    def __init__(self):
        params = {
            "objective":        "regression",
            "metric":           "mae",
            "num_leaves":       31,
            "learning_rate":    0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":     5,
            "verbose":          -1,
        }
        self.model_home  = lgb.LGBMRegressor(**params, n_estimators=300)
        self.model_away  = lgb.LGBMRegressor(**params, n_estimators=300)
        self.scaler      = StandardScaler()
        self.diff_cols   = None
        self._team_feat  = None   # guardado para predict()

    def fit(self, dataset: pd.DataFrame):
        """Entrena con el dataset de partidos históricos."""
        self.diff_cols = [c for c in dataset.columns if c.startswith("diff_")]

        X      = dataset[self.diff_cols].fillna(0)
        y_home = dataset["home_sets"]
        y_away = dataset["away_sets"]

        X_scaled = self.scaler.fit_transform(X)

        X_tr, X_val, yh_tr, yh_val, ya_tr, ya_val = train_test_split(
            X_scaled, y_home, y_away, test_size=0.15, random_state=42
        )

        self.model_home.fit(X_tr, yh_tr, eval_set=[(X_val, yh_val)])
        self.model_away.fit(X_tr, ya_tr, eval_set=[(X_val, ya_val)])

        pred_home = self.model_home.predict(X_val)
        pred_away = self.model_away.predict(X_val)
        print(f"  MAE home sets: {mean_absolute_error(yh_val, pred_home):.3f}")
        print(f"  MAE away sets: {mean_absolute_error(ya_val, pred_away):.3f}")

    def _store_team_features(self, df_teams, df_player_features):
        """Guarda internamente el DataFrame de features para usar en predict()."""
        team_feat = build_team_features(df_teams)
        if not df_player_features.empty:
            team_feat = team_feat.merge(df_player_features, on=["club", "season"], how="left")
        self._team_feat = team_feat

    def predict(
        self,
        home_club: str,
        away_club: str,
        season: str,
    ) -> dict:
        """
        Predice el resultado de un partido.
        Llama a _store_team_features() antes de usar este método.

        Returns:
            {
              "home_sets":         float,  # valor continuo predicho
              "away_sets":         float,
              "home_sets_rounded": int,    # redondeado y clampado a [0, 3]
              "away_sets_rounded": int,
            }
        """
        if self._team_feat is None:
            raise RuntimeError("Llama a _store_team_features() antes de predict().")

        home_f = get_team_vector(self._team_feat, normalize_club(home_club), season)
        away_f = get_team_vector(self._team_feat, normalize_club(away_club), season)
        diff   = (home_f - away_f).add_prefix("diff_")

        X = pd.DataFrame([diff])[self.diff_cols].fillna(0)
        X_scaled = self.scaler.transform(X)

        home_pred = float(self.model_home.predict(X_scaled)[0])
        away_pred = float(self.model_away.predict(X_scaled)[0])

        home_rounded = int(np.clip(round(home_pred), 0, 3))
        away_rounded = int(np.clip(round(away_pred), 0, 3))

        return {
            "home_sets":         round(home_pred, 2),
            "away_sets":         round(away_pred, 2),
            "home_sets_rounded": home_rounded,
            "away_sets_rounded": away_rounded,
        }

    def feature_importance(self) -> pd.DataFrame:
        """Devuelve la importancia de features del modelo local (home)."""
        imp = pd.DataFrame({
            "feature":    self.diff_cols,
            "importance": self.model_home.feature_importances_,
        }).sort_values("importance", ascending=False)
        return imp


# ─────────────────────────────────────────────
# 6. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def drop_empty_features(df):
    """Elimina columnas que no tienen ningún dato para que no rompan el dropna."""
    nan_pct = df.isna().mean() * 100
    cols_to_drop = nan_pct[nan_pct == 100].index.tolist()
    if cols_to_drop:
        print(f"\nEliminando columnas sin datos: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df

if __name__ == "__main__":

    # ── Rutas ──────────────────────────────────
    TEAM_FILE    = "DB/Comparacion_equipos_10_años.csv"
    PLAYER_FILES = glob.glob("DB/stats_por_equipo_completo/*_historial_10_años.csv")   # los 22 archivos de clubes
    MATCH_FILES  = glob.glob("DB/enfrentamientos_directos/enfrentamientos_directos_*.csv")  # todos los de temporadas

    print(f"Archivos de jugadores encontrados: {len(PLAYER_FILES)}")
    print(f"Archivos de partidos encontrados:  {len(MATCH_FILES)}")

    # ── Carga ──────────────────────────────────
    df_teams   = load_team_data(TEAM_FILE)
    df_players = load_player_data(PLAYER_FILES)
    matches    = load_matches(MATCH_FILES)

    print(f"\nEquipos × temporadas: {len(df_teams)}")
    print(f"Registros de jugadores: {len(df_players)}")
    print(f"Partidos totales: {len(matches)}")

    # ── Features de jugadores ──────────────────
    df_player_feat = build_player_features(df_players)

    # ── Dataset de partidos ────────────────────
    dataset = build_match_dataset(df_teams, df_player_feat, matches)
    print(f"\nDataset generado: {dataset.shape[0]} partidos × {dataset.shape[1]} columnas")
    print(f"NaNs por columna (%):\n"
          f"{(dataset[dataset.columns[dataset.columns.str.startswith('diff_')]].isna().mean()*100).round(1)}")
    
    dataset = drop_empty_features(dataset)

    # ── Entrenamiento ──────────────────────────
    print("\nEntrenando modelo...")
    model = SetPredictor()
    model.fit(dataset)
    model._store_team_features(df_teams, df_player_feat)

    # ── Importancia de features ────────────────
    print("\nTop 10 features más importantes:")
    print(model.feature_importance().head(10).to_string(index=False))

    # ── Predicción de un partido nuevo ─────────
    print("\n=== PREDICCIÓN DE EJEMPLO ===")
    result = model.predict(
        home_club="Grottazzolina",  # Usa el nombre corto que está en tu Excel de 10 años
        away_club="Perugia",
        season="2024/2025",
    )
    print(f"Grottazzolina: {result['home_sets_rounded']} sets  ({result['home_sets']})")
    print(f"Perugia:         {result['away_sets_rounded']} sets  ({result['away_sets']})")


    # --- LÍNEAS DE DIAGNÓSTICO ---
    print(f"Ejemplo nombres en df_teams: {df_teams['club'].unique()[:5]}")
    print(f"Ejemplo temporadas en df_teams: {df_teams['season'].unique()[:5]}")
    print(f"Ejemplo nombres en matches: {matches['home_club'].unique()[:5]}")
    print(f"Ejemplo temporadas en matches: {matches['season'].unique()[:5]}")

    # Verificar si hay filas sin NaNs
    non_nan_rows = dataset.dropna(subset=[c for c in dataset.columns if 'diff_' in c])
    print(f"\nPartidos con datos completos encontrados: {len(non_nan_rows)} de {len(dataset)}")
    # -----------------------------