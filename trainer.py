"""
entrenar.py
───────────
Entrena DOS modelos por cada temporada de corte y guarda un .pkl por corte:

  1. MatchPredictor  → predice el resultado global del partido (sets totales)
  2. SetPredictor    → predice el ganador y marcador de cada set individual

Lógica acumulativa (sin data leakage):
  modelo_hasta_2021_2022.pkl  →  datos hasta 2021/22
  modelo_hasta_2022_2023.pkl  →  datos hasta 2022/23
  ...
  modelo_hasta_2025_2026.pkl  →  todos los datos disponibles

Fuentes de datos:
  - DB/Comparacion_equipos_10_años.csv         (stats de equipo por temporada)
  - DB/stats_por_equipo_completo/*.csv          (stats de jugadores)
  - DB/enfrentamientos_directos/*.csv           (resultados de partidos)
  - DB/sets_partidos.csv                        (resultado set a set)

Uso:
    python entrenar.py
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════

TEAM_FILE    = "DB/Comparacion_equipos_10_años.csv"
PLAYER_FILES = glob.glob("DB/stats_por_equipo_completo/*_historial_10_años.csv")
MATCH_FILES  = glob.glob("DB/enfrentamientos_directos/enfrentamientos_directos_*.csv")
SETS_FILE    = "DB/sets_partidos.csv"
MODEL_DIR    = "model"

SEASONS_CUTOFF = [
    "2021/2022",
    "2022/2023",
    "2023/2024",
    "2024/2025",
    "2025/2026",
]


# ═════════════════════════════════════════════════════════════════
# 0. NORMALIZACIÓN DE NOMBRES
# ═════════════════════════════════════════════════════════════════

# Mapa genérico: nombre completo → nombre canónico
CLUB_NAME_MAP = {
    # Fuente: enfrentamientos_directos + legavolley scraper
    "Lube Civitanova":                      "Lube",
    "LubeCivitanova":                       "Lube",
    "Cucine Lube Civitanova":               "Lube",
    "Trentino":                             "Trento",
    "Diatec Trentino":                      "Trento",
    "Itas Trentino":                        "Trento",
    "Trentino Volley":                      "Trento",
    "Sir Safety Perugia":                   "Perugia",
    "Sir Safety Conad Perugia":             "Perugia",
    "Sir Safety Susa Perugia":              "Perugia",
    "Sir Susa Vim Perugia":                 "Perugia",
    "Sir Susa Scai Perugia":               "Perugia",
    "Vero Volley Monza":                    "Monza",
    "Mint Vero Volley Monza":               "Monza",
    "Gi Group Monza":                       "Monza",
    "Allianz Milano":                       "Milano",
    "Powervolley Milano":                   "Milano",
    "Revivre Milano":                       "Milano",
    "Revivre Axopower Milano":              "Milano",
    "Piacenza Copra":                       "Piacenza",
    "Gas Sales Piacenza":                   "Piacenza",
    "Gas Sales Bluenergy Piacenza":         "Piacenza",
    "Cuneo Volley":                         "Cuneo",
    "BAM Acqua S. Bernardo Cuneo":          "Cuneo",
    "Banca Alpi Marittime Acqua S.Bernardo Cuneo": "Cuneo",
    "Puliservice Acqua S.Bernardo Cuneo":   "Cuneo",
    "MA Acqua S.Bernardo Cuneo":            "Cuneo",
    "Vibo Valentia":                        "Vibo Valentia",
    "Castellana Grotte":                    "Castellana Grotte New Mater",
    "Acicastello":                          "Acicastello",
    "Saturnia Acicastello":                 "Acicastello",
    "Cosedil Acicastello":                  "Acicastello",
    "Sistemia Aci Castello":                "Acicastello",
    "Sistemia LCT Aci Castello":            "Acicastello",
    "Farmitalia Catania":                   "Acicastello",
    "Sviluppo Sud Catania":                 "Acicastello",
    "Azimut Modena":                        "Modena",
    "Azimut Leo Shoes Modena":              "Modena",
    "Leo Shoes PerkinElmer Modena":         "Modena",
    "Valsa Group Modena":                   "Modena",
    "Leo Shoes Modena":                     "Modena",
    "Kioene Padova":                        "Padova",
    "Sonepar Padova":                       "Padova",
    "Pallavolo Padova":                     "Padova",
    "Rana Verona":                          "Verona",
    "WithU Verona":                         "Verona",
    "Verona Volley":                        "Verona",
    "NBV Verona":                           "Verona",
    "Calzedonia Verona":                    "Verona",
    "Consar Ravenna":                       "Ravenna",
    "Consar RCM Ravenna":                   "Ravenna",
    "Emma Villas Siena":                    "Siena",
    "Emma Villas Aubay Siena":              "Siena",
    "Emma Villas Codyeco Lupi Siena":       "Siena",
    "Gioiella Prisma Taranto":              "Taranto",
    "Prisma Taranto":                       "Taranto",
    "Prisma La Cascina Taranto":            "Taranto",
    "Videx Grottazzolina":                  "Grottazzolina",
    "Videx Yuasa Grottazzolina":            "Grottazzolina",
    "Yuasa Battery Grottazzolina":          "Grottazzolina",
    "Pool Libertas Cantù":                  "Cantù",
    "Campi Reali Cantù":                    "Cantù",
    "Centrale del Latte McDonald's Brescia": "Brescia",
    "Centrale Del Latte Mcdonald'S Brescia": "Brescia",
    "Centrale del Latte Sferc Brescia":     "Brescia",
    "Consoli McDonald's Brescia":           "Brescia",
    "Consoli Sferc Brescia":               "Brescia",
    "Sarca Italia Chef Centrale Brescia":   "Brescia",
    "Gruppo Consoli Centrale del Latte Brescia": "Brescia",
    "Gruppo Consoli McDonald's Brescia":    "Brescia",
    "Gruppo Consoli Sferc Brescia":         "Brescia",
    "Tinet Gori Wines Prata di Pordenone":  "Prata di Pordenone",
    "Tinet Prata di Pordenone":             "Prata di Pordenone",
    "Biscottificio Marini Delta Po Porto Viro": "Porto Viro",
    "Delta Group Porto Viro":               "Porto Viro",
    "Delta Group Rico Carni Porto Viro":    "Porto Viro",
    "Alva Inox 2 Emme Service Porto Viro":  "Porto Viro",
    "Med Store Macerata":                   "Macerata",
    "Med Store Tunit Macerata":             "Macerata",
    "Menghi Shoes Macerata":               "Macerata",
    "Banca Macerata":                       "Macerata",
    "Banca Macerata Fisiomed MC":           "Macerata",
    "Gibam Fano":                           "Fano",
    "Smartsystem Fano":                     "Fano",
    "Vigilar Fano":                         "Fano",
    "Smartsystem Essence Hotels Fano":      "Fano",
    "Essence Hotels Fano":                  "Fano",
    "Shedirpharma Sorrento":               "Sorrento",
    "Shedirpharma Massa Lubrense":          "Sorrento",
    "Romeo Sorrento":                       "Sorrento",
    "WOW Green House Aversa":               "Aversa",
    "Wow Green House Aversa":               "Aversa",
    "Evolution Green Aversa":               "Aversa",
    "Sigma Aversa":                         "Aversa",
    "Normanna Aversa Academy":              "Aversa",
    "Virtus Aversa":                        "Aversa",
    "Basi Grafiche Geosat Lagonegro":       "Lagonegro",
    "Cave Del Sole Lagonegro":              "Lagonegro",
    "Cave del Sole Geomedical Lagonegro":   "Lagonegro",
    "Geosat Geovertical Lagonegro":         "Lagonegro",
    "Geovertical Geosat Lagonegro":         "Lagonegro",
    "Rinascita Lagonegro":                  "Lagonegro",
    "Abba Pineto":                          "Pineto",
}

# Alias que dependen de la temporada
CLUB_SEASON_MAP = {
    ("Cisterna", "2020/2021"): "Cisterna Top Volley",
    ("Cisterna", "2021/2022"): "Cisterna Top Volley",
    ("Cisterna", "2022/2023"): "Cisterna Top Volley",
}

def normalize_club(name: str, season: str = None) -> str:
    if not isinstance(name, str):
        return name
    s = name.strip()
    # Eliminar nombres duplicados (ej: "LubeLube")
    mid = len(s) // 2
    if len(s) > 4 and s[:mid] == s[mid:]:
        s = s[:mid]
    s = CLUB_NAME_MAP.get(s, s)
    if season is not None:
        s = CLUB_SEASON_MAP.get((s, season), s)
    return s


# ═════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═════════════════════════════════════════════════════════════════

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
            filename  = os.path.basename(p)
            team_name = filename.replace("_historial_10_años.csv", "").replace("_", " ")
            df["ID_Equipo"] = team_name
            frames.append(df)
        except Exception as e:
            print(f"  Error cargando {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_matches(paths: list) -> pd.DataFrame:
    frames = []
    for p in paths:
        df        = pd.read_csv(p)
        fname     = os.path.basename(p)
        year_code = fname.replace("enfrentamientos_directos_", "").replace(".csv", "")
        if len(year_code) == 8 and year_code.isdigit():
            df["season"] = f"{year_code[:4]}/{year_code[4:]}"
        df["home_club"] = df.apply(
            lambda r: normalize_club(r["home_club"], r["season"]), axis=1
        )
        df["away_club"] = df.apply(
            lambda r: normalize_club(r["away_club"], r["season"]), axis=1
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_sets_data(path: str) -> pd.DataFrame:
    """Carga el CSV de resultados set a set y normaliza nombres."""
    df = pd.read_csv(path)
    # Eliminar filas con nombres vacíos o inválidos
    df = df[df["equipo_local"].str.strip().str.len() > 1].copy()
    df = df[df["equipo_visitante"].str.strip().str.len() > 1].copy()
    df["equipo_local"]     = df.apply(
        lambda r: normalize_club(r["equipo_local"], r["temporada"]), axis=1
    )
    df["equipo_visitante"] = df.apply(
        lambda r: normalize_club(r["equipo_visitante"], r["temporada"]), axis=1
    )
    return df


# ═════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — STATS DE EQUIPO
# ═════════════════════════════════════════════════════════════════

def build_team_features(df_teams: pd.DataFrame) -> pd.DataFrame:
    df = df_teams.copy()
    sp = df["sets_played"].replace(0, np.nan)
    df["serve_ace_rate"]   = df["serve_ace"]   / sp
    df["att_err_rate"]     = df["att_err"]     / sp
    df["block_rate"]       = df["block_exc"]   / sp
    df["rec_err_rate"]     = df["rec_err"]     / sp
    df["att_blocked_rate"] = df["att_blocked"] / sp
    df["attack_pressure"]  = df["att_effic"] - df["rec_err_rate"]
    df["bp_ratio"]         = df["points_bp"] / df["points_tot"].replace(0, np.nan)
    return df


def build_player_features(df_players: pd.DataFrame) -> pd.DataFrame:
    if df_players.empty:
        return pd.DataFrame()
    df = df_players.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Played Set_Played Set": "sets_played",
        "ID_Equipo":             "club",
        "Temporada":             "season",
        "ATTACK_Effic.":         "att_effic",
        "ATTACK_Exc. %":         "att_exc_pct",
        "SERVE_Ace per Set":     "serve_ace_per_set",
        "SERVE_Effic.":          "serve_effic",
        "RECEPTION_Effic.":      "rec_effic",
        "RECEPTION_Exc. %":      "rec_exc_pct",
        "BLOCK_Points per Set":  "block_per_set",
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
    row = team_feat[
        (team_feat["club"] == club) & (team_feat["season"] == season)
    ]
    if row.empty:
        return pd.Series({c: np.nan for c in FEATURE_COLS})
    available = [c for c in FEATURE_COLS if c in row.columns]
    result    = pd.Series({c: np.nan for c in FEATURE_COLS})
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
    cols = [c for c in df.columns if df[c].isna().all()]
    if cols:
        print(f"  Eliminando columnas vacías: {cols}")
        df = df.drop(columns=cols)
    return df


# ═════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING — SETS
# ═════════════════════════════════════════════════════════════════

def build_pair_history(df_sets: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada par (local, visitante) calcula stats históricas:
      - % sets ganados por el local en ese enfrentamiento
      - % sets largos (>25 pts) en ese enfrentamiento
      - nº total de sets jugados entre ellos (volumen de historial)
    """
    records = []
    for (local, visitante), g in df_sets.groupby(["equipo_local", "equipo_visitante"]):
        records.append({
            "equipo_local":      local,
            "equipo_visitante":  visitante,
            "hist_pct_sets_local": g["ganador_set_local"].mean(),
            "hist_pct_set_largo":  g["set_largo"].mean(),
            "hist_n_sets":         len(g),
        })
    return pd.DataFrame(records)


def build_set_dataset(
    df_sets: pd.DataFrame,
    team_feat: pd.DataFrame,
    pair_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el dataset de entrenamiento del SetPredictor (modelo de sets).
    Una fila = un set jugado.

    Features:
      [A] Diferencial de stats de equipo (mismo bloque que MatchPredictor)
      [B] Contexto del set: número de set, marcador parcial, flags
      [C] Historial del par de equipos
    Target: ganador_set_local (0/1)
    """
    # Merge con historial del par
    df = df_sets.merge(pair_history, on=["equipo_local", "equipo_visitante"], how="left")

    records = []
    for _, row in df.iterrows():
        local     = row["equipo_local"]
        visitante = row["equipo_visitante"]
        season    = row["temporada"]
        set_num   = row["set_num"]

        # [A] Stats diferenciales de equipo
        home_v = get_team_vector(team_feat, local, season)
        away_v = get_team_vector(team_feat, visitante, season)
        diff   = (home_v - away_v).add_prefix("diff_").to_dict()

        # [B] Contexto del set
        # Reconstruir marcador parcial antes de este set
        # (en el CSV, set_num indica el set actual)
        sets_local_antes     = 0
        sets_visitante_antes = 0
        resultado            = row.get("resultado_final", "")
        if isinstance(resultado, str) and "-" in resultado:
            try:
                sl, sv = map(int, resultado.split("-"))
                # Distribuir sets anteriores de forma aproximada
                # (no tenemos el marcador acumulado por set, solo el final)
                # Usamos el número de set como proxy
                total_sets = sl + sv
                if set_num <= total_sets:
                    ratio = (set_num - 1) / max(total_sets, 1)
                    sets_local_antes     = round(sl * ratio)
                    sets_visitante_antes = round(sv * ratio)
            except ValueError:
                pass

        local_puede_cerrar     = 1 if sets_local_antes == 2 else 0
        visitante_puede_cerrar = 1 if sets_visitante_antes == 2 else 0
        es_desempate           = 1 if set_num == 5 else 0

        context = {
            "set_num":                  set_num,
            "sets_local_antes":         sets_local_antes,
            "sets_visitante_antes":     sets_visitante_antes,
            "diff_parcial":             sets_local_antes - sets_visitante_antes,
            "local_puede_cerrar":       local_puede_cerrar,
            "visitante_puede_cerrar":   visitante_puede_cerrar,
            "es_desempate":             es_desempate,
        }

        # [C] Historial del par
        history = {
            "hist_pct_sets_local": row.get("hist_pct_sets_local", 0.5),
            "hist_pct_set_largo":  row.get("hist_pct_set_largo", 0.1),
            "hist_n_sets":         row.get("hist_n_sets", 0),
        }

        record = {**diff, **context, **history,
                  "ganador_set_local": row["ganador_set_local"],
                  "set_largo":         row["set_largo"],
                  "puntos_local":      row["puntos_local"],
                  "puntos_visitante":  row["puntos_visitante"],
                  "equipo_local":      local,
                  "equipo_visitante":  visitante,
                  "temporada":         season,
                  "set_num_orig":      set_num}
        records.append(record)

    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════
# 4. MODELOS
# ═════════════════════════════════════════════════════════════════

LGB_PARAMS = {
    "num_leaves": 31, "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "verbose": -1,
}


class MatchPredictor:
    """Predice el nº de sets ganados por local y visitante (regresión)."""

    def __init__(self):
        self.model_home = lgb.LGBMRegressor(
            **LGB_PARAMS, objective="regression", metric="mae", n_estimators=300
        )
        self.model_away = lgb.LGBMRegressor(
            **LGB_PARAMS, objective="regression", metric="mae", n_estimators=300
        )
        self.scaler    = StandardScaler()
        self.feat_cols = None
        self._team_feat = None

    def fit(self, dataset: pd.DataFrame):
        self.feat_cols = [c for c in dataset.columns if c.startswith("diff_")]
        X      = dataset[self.feat_cols].fillna(0)
        y_home = dataset["home_sets"]
        y_away = dataset["away_sets"]
        Xs     = self.scaler.fit_transform(X)
        Xtr, Xval, yh_tr, yh_val, ya_tr, ya_val = train_test_split(
            Xs, y_home, y_away, test_size=0.15, random_state=42
        )
        self.model_home.fit(Xtr, yh_tr, eval_set=[(Xval, yh_val)])
        self.model_away.fit(Xtr, ya_tr, eval_set=[(Xval, ya_val)])
        mae_h = mean_absolute_error(yh_val, self.model_home.predict(Xval))
        mae_a = mean_absolute_error(ya_val, self.model_away.predict(Xval))
        print(f"  [MatchPredictor] MAE local: {mae_h:.3f}  |  MAE visitante: {mae_a:.3f}")

    def store_team_features(self, df_teams, df_player_features):
        tf = build_team_features(df_teams)
        if not df_player_features.empty:
            tf = tf.merge(df_player_features, on=["club", "season"], how="left")
        self._team_feat = tf

    def predict(self, home_club: str, away_club: str, season: str) -> dict:
        home_club = normalize_club(home_club, season)
        away_club = normalize_club(away_club, season)
        hv = get_team_vector(self._team_feat, home_club, season)
        av = get_team_vector(self._team_feat, away_club, season)
        diff = (hv - av).add_prefix("diff_")
        X  = pd.DataFrame([diff])[self.feat_cols].fillna(0)
        Xs = self.scaler.transform(X)
        hp = float(self.model_home.predict(Xs)[0])
        ap = float(self.model_away.predict(Xs)[0])
        return {
            "home_sets":         round(hp, 2),
            "away_sets":         round(ap, 2),
            "home_sets_rounded": int(np.clip(round(hp), 0, 3)),
            "away_sets_rounded": int(np.clip(round(ap), 0, 3)),
        }

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature":    self.feat_cols,
            "importance": self.model_home.feature_importances_,
        }).sort_values("importance", ascending=False)


class SetBySetPredictor:
    """
    Predice el ganador de cada set individual (clasificación binaria)
    y el marcador estimado (regresión de puntos).
    """

    def __init__(self):
        # Clasificador: ¿quién gana el set?
        self.model_winner = lgb.LGBMClassifier(
            **LGB_PARAMS, objective="binary", metric="binary_logloss",
            n_estimators=300
        )
        # Regresores: puntos del local y visitante en el set
        self.model_pts_local = lgb.LGBMRegressor(
            **LGB_PARAMS, objective="regression", metric="mae", n_estimators=300
        )
        self.model_pts_visitante = lgb.LGBMRegressor(
            **LGB_PARAMS, objective="regression", metric="mae", n_estimators=300
        )
        self.scaler      = StandardScaler()
        self.feat_cols   = None
        self._team_feat  = None
        self._pair_hist  = None

    def fit(self, dataset: pd.DataFrame):
        feat_exclude = {
            "ganador_set_local", "set_largo",
            "puntos_local", "puntos_visitante",
            "equipo_local", "equipo_visitante",
            "temporada", "set_num_orig",
        }
        self.feat_cols = [
            c for c in dataset.columns if c not in feat_exclude
        ]
        X  = dataset[self.feat_cols].fillna(0)
        yw = dataset["ganador_set_local"]
        yl = dataset["puntos_local"]
        ya = dataset["puntos_visitante"]
        Xs = self.scaler.fit_transform(X)

        Xtr, Xval, yw_tr, yw_val, yl_tr, yl_val, ya_tr, ya_val = train_test_split(
            Xs, yw, yl, ya, test_size=0.15, random_state=42
        )
        self.model_winner.fit(Xtr, yw_tr, eval_set=[(Xval, yw_val)])
        self.model_pts_local.fit(Xtr, yl_tr, eval_set=[(Xval, yl_val)])
        self.model_pts_visitante.fit(Xtr, ya_tr, eval_set=[(Xval, ya_val)])

        acc  = accuracy_score(yw_val, self.model_winner.predict(Xval))
        mael = mean_absolute_error(yl_val, self.model_pts_local.predict(Xval))
        maev = mean_absolute_error(ya_val, self.model_pts_visitante.predict(Xval))
        print(f"  [SetPredictor]   Accuracy ganador: {acc:.3f}  |  "
              f"MAE pts local: {mael:.2f}  |  MAE pts visitante: {maev:.2f}")

    def store_features(self, team_feat: pd.DataFrame, pair_hist: pd.DataFrame):
        self._team_feat = team_feat
        self._pair_hist = pair_hist

    def predict_set(
        self,
        local: str,
        visitante: str,
        season: str,
        set_num: int,
        sets_local_antes: int,
        sets_visitante_antes: int,
    ) -> dict:
        """Predice un set individual dado el contexto actual del partido."""
        local     = normalize_club(local, season)
        visitante = normalize_club(visitante, season)

        hv   = get_team_vector(self._team_feat, local, season)
        av   = get_team_vector(self._team_feat, visitante, season)
        diff = (hv - av).add_prefix("diff_").to_dict()

        # Contexto
        context = {
            "set_num":                  set_num,
            "sets_local_antes":         sets_local_antes,
            "sets_visitante_antes":     sets_visitante_antes,
            "diff_parcial":             sets_local_antes - sets_visitante_antes,
            "local_puede_cerrar":       1 if sets_local_antes == 2 else 0,
            "visitante_puede_cerrar":   1 if sets_visitante_antes == 2 else 0,
            "es_desempate":             1 if set_num == 5 else 0,
        }

        # Historial del par
        ph = self._pair_hist
        par = ph[
            (ph["equipo_local"] == local) &
            (ph["equipo_visitante"] == visitante)
        ]
        history = {
            "hist_pct_sets_local": par["hist_pct_sets_local"].iloc[0] if len(par) else 0.5,
            "hist_pct_set_largo":  par["hist_pct_set_largo"].iloc[0]  if len(par) else 0.1,
            "hist_n_sets":         par["hist_n_sets"].iloc[0]          if len(par) else 0,
        }

        row    = {**diff, **context, **history}
        X      = pd.DataFrame([row])[self.feat_cols].fillna(0)
        Xs     = self.scaler.transform(X)

        prob_local = float(self.model_winner.predict_proba(Xs)[0][1])
        pts_local  = float(self.model_pts_local.predict(Xs)[0])
        pts_visit  = float(self.model_pts_visitante.predict(Xs)[0])

        # Ajustar marcador para que sea coherente con el ganador predicho
        # Si gana el local, sus puntos deben ser >= pts_visit + 1
        if prob_local >= 0.5:
            pts_local = max(pts_local, pts_visit + 1)
        else:
            pts_visit = max(pts_visit, pts_local + 1)

        margen = 4  # ±4 puntos de rango
        return {
            "prob_local":    round(prob_local, 3),
            "prob_visitante": round(1 - prob_local, 3),
            "gana_local":    prob_local >= 0.5,
            "pts_local_est": round(pts_local),
            "pts_visit_est": round(pts_visit),
            "pts_local_min": round(pts_local) - margen,
            "pts_local_max": round(pts_local) + margen,
            "pts_visit_min": round(pts_visit) - margen,
            "pts_visit_max": round(pts_visit) + margen,
        }

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature":    self.feat_cols,
            "importance": self.model_winner.feature_importances_,
        }).sort_values("importance", ascending=False)


class VBPredictor:
    """Contenedor que agrupa MatchPredictor + SetBySetPredictor en un único .pkl."""

    def __init__(self):
        self.match_model = MatchPredictor()
        self.set_model   = SetBySetPredictor()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"  ✅ Guardado: '{path}'  ({os.path.getsize(path)//1024} KB)")

    @classmethod
    def load(cls, path: str) -> "VBPredictor":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Modelo no encontrado: '{path}'\nEjecuta entrenar.py primero."
            )
        return joblib.load(path)


# ═════════════════════════════════════════════════════════════════
# 5. PIPELINE DE ENTRENAMIENTO
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  PREDICTOR VB — Entrenamiento completo")
    print("=" * 60)
    print(f"\n  Archivos jugadores : {len(PLAYER_FILES)}")
    print(f"  Archivos partidos  : {len(MATCH_FILES)}")

    # ── Carga completa (una sola vez) ─────────────────────────────
    df_teams_all       = load_team_data(TEAM_FILE)
    df_players_all     = load_player_data(PLAYER_FILES)
    matches_all        = load_matches(MATCH_FILES)
    df_player_feat_all = build_player_features(df_players_all)
    df_sets_all        = load_sets_data(SETS_FILE)

    print(f"\n  Equipos × temporadas : {len(df_teams_all)}")
    print(f"  Registros jugadores  : {len(df_players_all)}")
    print(f"  Partidos totales     : {len(matches_all)}")
    print(f"  Sets totales         : {len(df_sets_all)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    for cutoff in SEASONS_CUTOFF:
        print(f"\n{'─' * 60}")
        print(f"  Modelo hasta: {cutoff}")
        print(f"{'─' * 60}")

        seasons_ok = [
            s for s in sorted(matches_all["season"].unique()) if s <= cutoff
        ]
        print(f"  Temporadas: {seasons_ok}")

        # Filtrar por corte temporal
        matches_cut    = matches_all[matches_all["season"].isin(seasons_ok)]
        df_teams_cut   = df_teams_all[df_teams_all["season"].isin(seasons_ok)]
        df_player_feat = df_player_feat_all[df_player_feat_all["season"].isin(seasons_ok)]
        df_sets_cut    = df_sets_all[df_sets_all["temporada"].isin(seasons_ok)]

        print(f"  Partidos : {len(matches_cut)}  |  Sets: {len(df_sets_cut)}")

        # Features de equipo compartidas
        team_feat = build_team_features(df_teams_cut)
        if not df_player_feat.empty:
            team_feat = team_feat.merge(df_player_feat, on=["club", "season"], how="left")

        # ── MatchPredictor ─────────────────────────────────────────
        print(f"\n  → MatchPredictor")
        match_dataset = build_match_dataset(df_teams_cut, df_player_feat, matches_cut)
        match_dataset = drop_empty_features(match_dataset)
        diff_cols     = [c for c in match_dataset.columns if c.startswith("diff_")]
        completos     = match_dataset.dropna(subset=diff_cols)
        print(f"    Partidos completos: {len(completos)} / {len(match_dataset)}")

        if len(completos) < 20:
            print("    ⚠️  Muy pocos datos, saltando este corte.")
            continue

        vb = VBPredictor()
        vb.match_model.fit(match_dataset)
        vb.match_model.store_team_features(df_teams_cut, df_player_feat)

        top5 = vb.match_model.feature_importance().head(5)
        print("    Top 5 features:")
        for _, r in top5.iterrows():
            print(f"      {r['feature']:<45} {int(r['importance'])}")

        # ── SetBySetPredictor ──────────────────────────────────────
        print(f"\n  → SetBySetPredictor")
        pair_hist  = build_pair_history(df_sets_cut)
        set_dataset = build_set_dataset(df_sets_cut, team_feat, pair_hist)
        set_dataset = drop_empty_features(set_dataset)

        feat_excl = {
            "ganador_set_local", "set_largo", "puntos_local", "puntos_visitante",
            "equipo_local", "equipo_visitante", "temporada", "set_num_orig",
        }
        feat_cols_set = [c for c in set_dataset.columns if c not in feat_excl]
        completos_set = set_dataset.dropna(subset=feat_cols_set)
        print(f"    Sets completos: {len(completos_set)} / {len(set_dataset)}")

        if len(completos_set) < 50:
            print("    ⚠️  Muy pocos sets, saltando SetPredictor para este corte.")
        else:
            vb.set_model.fit(set_dataset)
            vb.set_model.store_features(team_feat, pair_hist)

            top5s = vb.set_model.feature_importance().head(5)
            print("    Top 5 features:")
            for _, r in top5s.iterrows():
                print(f"      {r['feature']:<45} {int(r['importance'])}")

        # ── Guardar ────────────────────────────────────────────────
        safe = cutoff.replace("/", "_")
        vb.save(os.path.join(MODEL_DIR, f"modelo_hasta_{safe}.pkl"))

    # ── Resumen final ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Modelos generados en model/:")
    for f in sorted(glob.glob(os.path.join(MODEL_DIR, "modelo_hasta_*.pkl"))):
        print(f"    {os.path.basename(f)}  ({os.path.getsize(f)//1024} KB)")
    print(f"\n  Usa predecir.py para hacer predicciones.")
    print("=" * 60)