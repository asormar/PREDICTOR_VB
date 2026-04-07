"""
entrenar.py
───────────
Entrena un modelo SetPredictor por cada temporada de corte y guarda un .pkl
independiente en la carpeta model/.

Lógica acumulativa (sin data leakage):
  - modelo_hasta_2021_2022.pkl  →  entrena con datos de 2020/21 + 2021/22
  - modelo_hasta_2022_2023.pkl  →  entrena con datos hasta 2022/23
  - ...
  - modelo_hasta_2025_2026.pkl  →  entrena con todos los datos disponibles

Al predecir con un modelo X, solo se usan las stats de temporadas ≤ X,
eliminando el data leakage de ver estadísticas futuras.
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════

TEAM_FILE    = "DB/Comparacion_equipos_10_años.csv"
PLAYER_FILES = glob.glob("DB/stats_por_equipo_completo/*_historial_10_años.csv")
MATCH_FILES  = glob.glob("DB/enfrentamientos_directos/enfrentamientos_directos_*.csv")
MODEL_DIR    = "model"

# Temporadas de corte: se genera un modelo por cada una (acumulativo hasta esa).
SEASONS_CUTOFF = [
    "2021/2022",
    "2022/2023",
    "2023/2024",
    "2024/2025",
    "2025/2026",
]


# ═════════════════════════════════════════════
# 0. NORMALIZACIÓN DE NOMBRES DE CLUBES
# ═════════════════════════════════════════════

CLUB_NAME_MAP = {
    "Lube Civitanova":      "Lube",
    "LubeCivitanova":       "Lube",
    "Trentino":             "Trento",
    "Sir Safety Perugia":   "Perugia",
    "Vero Volley Monza":    "Monza",
    "Allianz Milano":       "Milano",
    "Powervolley Milano":   "Milano",
    "Piacenza Copra":       "Piacenza",
    "Gas Sales Piacenza":   "Piacenza",
    "Cuneo Volley":         "Cuneo",
    "Vibo Valentia":        "Vibo Valentia",
    "Castellana Grotte":    "Castellana Grotte New Mater",
    "Acicastello":          "Acicastello",
    "Saturnia Acicastello": "Acicastello",
    "Sviluppo Sud Catania": "Acicastello",
}

# Equipos que cambiaron de nombre en temporadas concretas.
# {(nombre_en_matches, season): nombre_en_df_teams}
CLUB_SEASON_MAP = {
    ("Cisterna", "2020/2021"): "Cisterna Top Volley",
    ("Cisterna", "2021/2022"): "Cisterna Top Volley",
    ("Cisterna", "2022/2023"): "Cisterna Top Volley",
}

def normalize_club(name: str, season: str = None) -> str:
    if not isinstance(name, str): return name
    s = name.strip()
    mid = len(s) // 2
    if len(s) > 4 and s[:mid] == s[mid:]:
        s = s[:mid]
    s = CLUB_NAME_MAP.get(s, s)
    if season is not None:
        s = CLUB_SEASON_MAP.get((s, season), s)
    return s


# ═════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═════════════════════════════════════════════

def load_team_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Club_Club":                     "club",
        "Played Matches_Played Matches": "matches",
        "Played Set_Played Set":         "sets_played",
        "POINTS_Tot":                    "points_tot",
        "POINTS_BP":                     "points_bp",
        "SERVE_Tot":                     "serve_tot",
        "SERVE_Ace":                     "serve_ace",
        "SERVE_Err.":                    "serve_err",
        "SERVE_Ace per Set":             "serve_ace_per_set",
        "SERVE_Effic.":                  "serve_effic",
        "RECEPTION_Tot":                 "rec_tot",
        "RECEPTION_Err.":                "rec_err",
        "RECEPTION_Neg.":                "rec_neg",
        "RECEPTION_Exc.":                "rec_exc",
        "RECEPTION_Exc. %":              "rec_exc_pct",
        "RECEPTION_Effic.":              "rec_effic",
        "ATTACK_Tot":                    "att_tot",
        "ATTACK_Err.":                   "att_err",
        "ATTACK_Blocked":                "att_blocked",
        "ATTACK_Exc.":                   "att_exc",
        "ATTACK_Exc. %":                 "att_exc_pct",
        "ATTACK_Effic.":                 "att_effic",
        "BLOCK_Exc.":                    "block_exc",
        "BLOCK_Points per Set":          "block_per_set",
        "Temporada":                     "season",
    })
    def fix_season(x):
        if isinstance(x, (int, float, np.integer)):
            return f"{int(x)}/{int(x)+1}"
        return str(x)
    df["season"] = df["season"].apply(fix_season)
    df["club"]   = df["club"].apply(normalize_club)
    return df


def load_player_data(paths: list) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            filename = os.path.basename(p)
            team_name = filename.replace("_historial_10_años.csv", "").replace("_", " ")
            df["ID_Equipo"] = team_name
            frames.append(df)
        except Exception as e:
            print(f"  Error cargando {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_matches(paths: list) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        fname     = os.path.basename(p)
        year_code = fname.replace("enfrentamientos_directos_", "").replace(".csv", "")
        if len(year_code) == 8 and year_code.isdigit():
            df["season"] = f"{year_code[:4]}/{year_code[4:]}"
        df["home_club"] = df.apply(lambda r: normalize_club(r["home_club"], r["season"]), axis=1)
        df["away_club"] = df.apply(lambda r: normalize_club(r["away_club"], r["season"]), axis=1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ═════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═════════════════════════════════════════════

def build_team_features(df_teams: pd.DataFrame) -> pd.DataFrame:
    df = df_teams.copy()
    sp = df["sets_played"].replace(0, np.nan)
    df["serve_ace_rate"]   = df["serve_ace"]    / sp
    df["att_err_rate"]     = df["att_err"]      / sp
    df["block_rate"]       = df["block_exc"]    / sp
    df["rec_err_rate"]     = df["rec_err"]      / sp
    df["att_blocked_rate"] = df["att_blocked"]  / sp
    df["attack_pressure"]  = df["att_effic"] - df["rec_err_rate"]
    df["bp_ratio"]         = df["points_bp"] / df["points_tot"].replace(0, np.nan)
    return df


def build_player_features(df_players: pd.DataFrame) -> pd.DataFrame:
    if df_players.empty: return pd.DataFrame()
    df = df_players.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Played Set_Played Set":  "sets_played",
        "ID_Equipo":              "club",
        "Temporada":              "season",
        "ATTACK_Effic.":          "att_effic",
        "ATTACK_Exc. %":          "att_exc_pct",
        "SERVE_Ace per Set":      "serve_ace_per_set",
        "SERVE_Effic.":           "serve_effic",
        "RECEPTION_Effic.":       "rec_effic",
        "RECEPTION_Exc. %":       "rec_exc_pct",
        "BLOCK_Points per Set":   "block_per_set",
    })
    def fix_season(x):
        x = str(x)
        return x if "/" in x else f"{x}/{int(x)+1}"
    df["season"] = df["season"].apply(fix_season)
    df["club"]   = df.apply(lambda r: normalize_club(r["club"], r["season"]), axis=1)
    df["weight"] = pd.to_numeric(df["sets_played"], errors="coerce").fillna(0)

    PLAYER_COLS = [
        "att_effic", "att_exc_pct",
        "serve_ace_per_set", "serve_effic",
        "rec_effic", "rec_exc_pct",
        "block_per_set",
    ]
    records = []
    for (club, season), g in df.groupby(["club", "season"]):
        rec = {"club": club, "season": season}
        for col in PLAYER_COLS:
            if col in g.columns:
                w = g["weight"]
                v = pd.to_numeric(g[col], errors="coerce").fillna(0)
                rec[f"player_{col}_wmean"] = (v * w).sum() / w.sum() if w.sum() > 0 else 0
        records.append(rec)
    return pd.DataFrame(records)


FEATURE_COLS = [
    "serve_ace_rate", "serve_effic",
    "rec_exc_pct", "rec_effic", "rec_err_rate",
    "att_exc_pct", "att_effic", "att_err_rate", "att_blocked_rate",
    "block_rate", "block_per_set",
    "attack_pressure", "bp_ratio",
    "player_att_effic_wmean",
    "player_att_exc_pct_wmean",
    "player_serve_ace_per_set_wmean",
    "player_serve_effic_wmean",
    "player_rec_effic_wmean",
    "player_rec_exc_pct_wmean",
    "player_block_per_set_wmean",
]


def get_team_vector(team_feat: pd.DataFrame, club: str, season: str) -> pd.Series:
    row = team_feat[(team_feat["club"] == club) & (team_feat["season"] == season)]
    if row.empty:
        return pd.Series({c: np.nan for c in FEATURE_COLS})
    available = [c for c in FEATURE_COLS if c in row.columns]
    result = pd.Series({c: np.nan for c in FEATURE_COLS})
    result.update(row[available].iloc[0])
    return result


def build_match_dataset(df_teams, df_player_features, matches_df) -> pd.DataFrame:
    team_feat = build_team_features(df_teams)
    if not df_player_features.empty:
        team_feat = team_feat.merge(df_player_features, on=["club", "season"], how="left")
    records = []
    for _, match in matches_df.iterrows():
        home_f = get_team_vector(team_feat, match["home_club"], match["season"])
        away_f = get_team_vector(team_feat, match["away_club"], match["season"])
        record = (home_f - away_f).add_prefix("diff_").to_dict()
        record.update({
            "season":    match["season"],
            "home_club": match["home_club"],
            "away_club": match["away_club"],
            "home_sets": match["home_sets"],
            "away_sets": match["away_sets"],
        })
        records.append(record)
    return pd.DataFrame(records)


def drop_empty_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in df.columns if df[c].isna().all()]
    if cols_to_drop:
        print(f"  Eliminando columnas vacías: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


# ═════════════════════════════════════════════
# 3. MODELO
# ═════════════════════════════════════════════

class SetPredictor:
    """Predice sets ganados por el equipo local y visitante (dos LightGBM)."""

    def __init__(self):
        params = {
            "objective": "regression", "metric": "mae",
            "num_leaves": 31, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "verbose": -1,
        }
        self.model_home = lgb.LGBMRegressor(**params, n_estimators=300)
        self.model_away = lgb.LGBMRegressor(**params, n_estimators=300)
        self.scaler     = StandardScaler()
        self.diff_cols  = None
        self._team_feat = None

    def fit(self, dataset: pd.DataFrame):
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
        print(f"  MAE home sets: {mean_absolute_error(yh_val, self.model_home.predict(X_val)):.3f}")
        print(f"  MAE away sets: {mean_absolute_error(ya_val, self.model_away.predict(X_val)):.3f}")

    def store_team_features(self, df_teams, df_player_features):
        team_feat = build_team_features(df_teams)
        if not df_player_features.empty:
            team_feat = team_feat.merge(df_player_features, on=["club", "season"], how="left")
        self._team_feat = team_feat

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature":    self.diff_cols,
            "importance": self.model_home.feature_importances_,
        }).sort_values("importance", ascending=False)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"  ✅ Guardado en '{path}' ({os.path.getsize(path)/1024:.0f} KB)")

    @classmethod
    def load(cls, path: str) -> "SetPredictor":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado: '{path}'. Ejecuta entrenar.py primero.")
        return joblib.load(path)


# ═════════════════════════════════════════════
# 4. PIPELINE DE ENTRENAMIENTO
# ═════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 55)
    print("  PREDICTOR VB — Entrenamiento de modelos temporales")
    print("=" * 55)
    print(f"\n  Archivos de jugadores : {len(PLAYER_FILES)}")
    print(f"  Archivos de partidos  : {len(MATCH_FILES)}")

    # Carga completa una sola vez
    df_teams_all      = load_team_data(TEAM_FILE)
    df_players_all    = load_player_data(PLAYER_FILES)
    matches_all       = load_matches(MATCH_FILES)
    df_player_feat_all = build_player_features(df_players_all)

    print(f"\n  Equipos × temporadas  : {len(df_teams_all)}")
    print(f"  Registros jugadores   : {len(df_players_all)}")
    print(f"  Partidos totales      : {len(matches_all)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    for cutoff in SEASONS_CUTOFF:
        print(f"\n{'─' * 55}")
        print(f"  Modelo hasta: {cutoff}")
        print(f"{'─' * 55}")

        seasons_ok = [s for s in sorted(matches_all["season"].unique()) if s <= cutoff]
        print(f"  Temporadas incluidas : {seasons_ok}")

        matches_cut    = matches_all[matches_all["season"].isin(seasons_ok)]
        df_teams_cut   = df_teams_all[df_teams_all["season"].isin(seasons_ok)]
        df_player_feat = df_player_feat_all[df_player_feat_all["season"].isin(seasons_ok)]

        print(f"  Partidos             : {len(matches_cut)}")
        print(f"  Equipos × temporadas : {len(df_teams_cut)}")

        dataset = build_match_dataset(df_teams_cut, df_player_feat, matches_cut)
        dataset = drop_empty_features(dataset)

        diff_cols  = [c for c in dataset.columns if c.startswith("diff_")]
        completos  = dataset.dropna(subset=diff_cols)
        print(f"  Partidos completos   : {len(completos)} / {len(dataset)}")

        if len(completos) < 20:
            print("  ⚠️  Muy pocos datos, saltando.")
            continue

        model = SetPredictor()
        model.fit(dataset)
        model.store_team_features(df_teams_cut, df_player_feat)

        top5 = model.feature_importance().head(5)
        print("  Top 5 features:")
        for _, row in top5.iterrows():
            print(f"    {row['feature']:<45} {int(row['importance'])}")

        safe = cutoff.replace("/", "_")
        model.save(os.path.join(MODEL_DIR, f"modelo_hasta_{safe}.pkl"))

    print(f"\n{'=' * 55}")
    print("  Modelos generados en model/:")
    for f in sorted(glob.glob(os.path.join(MODEL_DIR, "modelo_hasta_*.pkl"))):
        print(f"    {os.path.basename(f)}  ({os.path.getsize(f)//1024} KB)")
    print(f"\n  Usa predecir.py para hacer predicciones.")
    print("=" * 55)