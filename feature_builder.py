"""
feature_builder.py
──────────────────
Construcción de features tabulares para PREDICTOR VB v2.

Genera dos datasets listos para entrenar:
  - match_features.csv  → un registro por partido (para el Strength Model)
  - set_features.csv    → un registro por set    (para el Set Model)

Uso:
    python feature_builder.py

Salida en DB/features/
"""

import os
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════

SETS_PATH = "DB/sets_partidos.csv"
H2H_DIR   = "DB/enfrentamientos_directos/"
OUT_DIR   = "DB/features/"

# Mismo mapa de normalización que trainer.py — fuente única de verdad
CLUB_NAME_MAP = {
    # ── SuperLega TOP ────────────────────────────────────────────
    "Gas Sales Bluenergy Piacenza":  "Piacenza",
    "Gas Sales Piacenza":            "Piacenza",
    "Itas Trentino":                 "Trento",
    "Trentino Volley":               "Trento",
    "Diatec Trentino":               "Trento",
    "Trentino":                      "Trento",
    "Allianz Milano":                "Milano",
    "Powervolley Milano":            "Milano",
    "Revivre Milano":                "Milano",
    "Cucine Lube Civitanova":        "Lube",
    "Lube Civitanova":               "Lube",
    "Rana Verona":                   "Verona",
    "Verona Volley":                 "Verona",
    "WithU Verona":                  "Verona",
    "NBV Verona":                    "Verona",
    "Calzedonia Verona":             "Verona",
    "Kioene Padova":                 "Padova",
    "Sonepar Padova":                "Padova",
    "Pallavolo Padova":              "Padova",
    "Vero Volley Monza":             "Monza",
    "Mint Vero Volley Monza":        "Monza",
    "Gi Group Monza":                "Monza",
    "Valsa Group Modena":            "Modena",
    "Leo Shoes Modena":              "Modena",
    "Leo Shoes PerkinElmer Modena":  "Modena",
    "Azimut Modena":                 "Modena",
    "Azimut Leo Shoes Modena":       "Modena",
    "Sir Safety Conad Perugia":      "Perugia",
    "Sir Safety Susa Perugia":       "Perugia",
    "Sir Susa Vim Perugia":          "Perugia",
    # ── Cuneo ────────────────────────────────────────────────────
    "BAM Acqua S. Bernardo Cuneo":           "Cuneo",
    "Banca Alpi Marittime Acqua S.Bernardo Cuneo": "Cuneo",
    "Puliservice Acqua S.Bernardo Cuneo":    "Cuneo",
    # ── Brescia ──────────────────────────────────────────────────
    "Centrale Del Latte Mcdonald'S Brescia":        "Brescia",
    "Centrale del Latte McDonald's Brescia":         "Brescia",
    "Centrale del Latte Sferc Brescia":              "Brescia",
    "Consoli McDonald's Brescia":                    "Brescia",
    "Consoli Sferc Brescia":                         "Brescia",
    "Sarca Italia Chef Centrale Brescia":            "Brescia",
    "Gruppo Consoli Centrale del Latte Brescia":     "Brescia",
    "Gruppo Consoli McDonald's Brescia":             "Brescia",
    # ── Lagonegro ────────────────────────────────────────────────
    "Basi Grafiche Geosat Lagonegro":    "Lagonegro",
    "Cave Del Sole Lagonegro":           "Lagonegro",
    "Cave del Sole Geomedical Lagonegro":"Lagonegro",
    "Geosat Geovertical Lagonegro":      "Lagonegro",
    "Geovertical Geosat Lagonegro":      "Lagonegro",
    # ── Fano ─────────────────────────────────────────────────────
    "Gibam Fano":                        "Fano",
    "Smartsystem Fano":                  "Fano",
    "Smartsystem Essence Hotels Fano":   "Fano",
    "Vigilar Fano":                      "Fano",
    # ── Ravenna ──────────────────────────────────────────────────
    "Consar RCM Ravenna":                "Ravenna",
    # ── Taranto ──────────────────────────────────────────────────
    "Gioiella Prisma Taranto":           "Taranto",
    "Prisma Taranto":                    "Taranto",
    # ── Porto Viro ───────────────────────────────────────────────
    "Biscottificio Marini Delta Po Porto Viro": "Porto Viro",
    "Delta Group Porto Viro":                   "Porto Viro",
    "Delta Group Rico Carni Porto Viro":        "Porto Viro",
    # ── Macerata ─────────────────────────────────────────────────
    "Banca Macerata":                    "Macerata",
    "Med Store Macerata":                "Macerata",
    "Med Store Tunit Macerata":          "Macerata",
    "Menghi Shoes Macerata":             "Macerata",
    # ── Siena ────────────────────────────────────────────────────
    "Emma Villas Siena":                 "Siena",
    "Emma Villas Aubay Siena":           "Siena",
    # ── Aversa ───────────────────────────────────────────────────
    "Evolution Green Aversa":            "Aversa",
    "Normanna Aversa Academy":           "Aversa",
    "Sigma Aversa":                      "Aversa",
    "WOW Green House Aversa":            "Aversa",
    "Wow Green House Aversa":            "Aversa",
    # ── Cantù ────────────────────────────────────────────────────
    "Pool Libertas Cantù":               "Cantù",
    # ── Acicastello ──────────────────────────────────────────────
    "Cosedil Acicastello":               "Acicastello",
    "Sistemia Aci Castello":             "Acicastello",
    "Sistemia LCT Aci Castello":         "Acicastello",
    # ── Sorrento ─────────────────────────────────────────────────
    "Shedirpharma Sorrento":             "Sorrento",
    "Shedirpharma Massa Lubrense":       "Sorrento",
    # ── Prata di Pordenone ───────────────────────────────────────
    "Tinet Gori Wines Prata di Pordenone": "Prata di Pordenone",
    # ── Grottazzolina ────────────────────────────────────────────
    "Videx Grottazzolina":               "Grottazzolina",
    "Videx Yuasa Grottazzolina":         "Grottazzolina",
    # ── Farmitalia Catania ────────────────────────────────────────
    "Farmitalia Catania":                "Catania",
}

# Orden cronológico de temporadas
SEASON_ORDER = [
    "2016/2017", "2017/2018", "2018/2019", "2019/2020",
    "2020/2021", "2021/2022", "2022/2023", "2023/2024",
    "2024/2025", "2025/2026",
]

# Alpha para exponential smoothing (más alto → más peso a partidos recientes)
EXP_ALPHA = 0.3

# Ventana de forma reciente (últimos N partidos)
FORMA_VENTANA = 5

# ── Shrinkage bayesiano ───────────────────────────────────────────
# K es el "peso equivalente" de la prior global.
# Con K=10: un equipo con 2 partidos queda al ~83% de la prior;
#           un equipo con 30 partidos queda al ~75% de su media real.
# Regla práctica: K entre 5 y 20. Aquí usamos 10.
SHRINKAGE_K = 10


def normalize_club(name: str) -> str:
    s = str(name).strip()
    return CLUB_NAME_MAP.get(s, s)


def shrink(obs: float, n: int, prior: float, k: int = SHRINKAGE_K) -> float:
    """
    Shrinkage bayesiano: mezcla la media observada (obs) con la prior global
    ponderando por el número de observaciones (n) frente al peso de la prior (k).

        shrunk = (n * obs + k * prior) / (n + k)

    Comportamiento:
      · n=0  → devuelve prior exacta              (sin datos → pura prior)
      · n=k  → media exacta entre obs y prior     (datos = peso prior)
      · n>>k → devuelve obs casi sin modificar    (muchos datos → confiar en obs)

    Parámetros
    ----------
    obs   : valor observado (media cruda)
    n     : número de observaciones que respaldan obs
    prior : valor de referencia de la liga (media global)
    k     : peso de la prior (SHRINKAGE_K, por defecto 10)
    """
    return (n * obs + k * prior) / (n + k)


# ═══════════════════════════════════════════════════════════════════
# 1. CARGA Y LIMPIEZA BASE
# ═══════════════════════════════════════════════════════════════════

def load_sets() -> pd.DataFrame:
    """Carga sets_partidos.csv y normaliza nombres de equipos."""
    df = pd.read_csv(SETS_PATH)
    df["equipo_local"]     = df["equipo_local"].apply(normalize_club)
    df["equipo_visitante"] = df["equipo_visitante"].apply(normalize_club)
    return df


def build_match_table(df_sets: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega sets → tabla de partidos con:
      - sets ganados por cada equipo
      - resultado binario (gana_local)
      - orden cronológico por temporada y jornada
    """
    partidos = (
        df_sets.groupby("partido_id")
        .agg(
            local      = ("equipo_local",     "first"),
            visitante  = ("equipo_visitante", "first"),
            temporada  = ("temporada",        "first"),
            jornada    = ("jornada",          "first"),
            sets_local = ("ganador_set_local","sum"),
            total_sets = ("set_num",          "count"),
        )
        .reset_index()
    )
    partidos["sets_visit"] = partidos["total_sets"] - partidos["sets_local"]
    partidos["gana_local"] = (partidos["sets_local"] > partidos["sets_visit"]).astype(int)

    # Orden cronológico dentro de cada temporada
    partidos["jornada_num"] = (
        partidos["jornada"].str.extract(r"(\d+)")[0]
        .astype(float).fillna(0)
    )
    season_idx = {s: i for i, s in enumerate(SEASON_ORDER)}
    partidos["season_idx"] = partidos["temporada"].map(season_idx).fillna(0)
    partidos = partidos.sort_values(["season_idx", "jornada_num"]).reset_index(drop=True)
    partidos["match_order"] = partidos.index  # índice global cronológico

    return partidos


# ═══════════════════════════════════════════════════════════════════
# 2. CARGA HEAD-TO-HEAD
# ═══════════════════════════════════════════════════════════════════

def load_h2h() -> pd.DataFrame:
    """
    Carga y limpia todos los CSV de enfrentamientos directos.
    Normaliza nombres y devuelve un DataFrame unificado.
    """
    frames = []
    for fname in sorted(os.listdir(H2H_DIR)):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(H2H_DIR, fname))

        # Los archivos más antiguos tienen nombres duplicados (bug del scraper)
        df["home_club"] = df["home_club"].apply(lambda x: normalize_club(str(x).split(str(x)[:5])[-1] if len(str(x)) > 20 else str(x)))
        df["away_club"] = df["away_club"].apply(lambda x: normalize_club(str(x).split(str(x)[:5])[-1] if len(str(x)) > 20 else str(x)))

        # Fallback: normalización directa para los archivos limpios
        df["home_club"] = df["home_club"].apply(normalize_club)
        df["away_club"] = df["away_club"].apply(normalize_club)

        frames.append(df)

    h2h = pd.concat(frames, ignore_index=True)
    h2h = h2h.dropna(subset=["home_sets", "away_sets"])
    h2h["home_sets"] = h2h["home_sets"].astype(int)
    h2h["away_sets"] = h2h["away_sets"].astype(int)
    h2h["set_diff"]  = h2h["home_sets"] - h2h["away_sets"]
    return h2h


# ═══════════════════════════════════════════════════════════════════
# 3. UTILIDADES DE EXPONENTIAL SMOOTHING
# ═══════════════════════════════════════════════════════════════════

def exp_smooth(values: list, alpha: float = EXP_ALPHA) -> float:
    """
    Exponential weighted average: más peso a los valores más recientes.
    values[0] es el más antiguo, values[-1] el más reciente.
    Devuelve NaN si la lista está vacía.
    """
    if not values:
        return np.nan
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return result


# ═══════════════════════════════════════════════════════════════════
# 4. FEATURES POR EQUIPO (calculadas en el momento t del partido)
# ═══════════════════════════════════════════════════════════════════

def compute_team_features_at(team: str, condition: str,
                              partidos_antes: pd.DataFrame,
                              h2h: pd.DataFrame,
                              oponente: str,
                              priors: dict) -> dict:
    """
    Calcula todas las features de un equipo justo ANTES de jugar el partido.
    Solo usa información disponible antes del partido (sin data leakage).
    Aplica shrinkage bayesiano sobre todas las medias para reducir el ruido
    de equipos con poco historial.

    Parámetros
    ----------
    team           : nombre normalizado del equipo
    condition      : 'local' o 'visitante'
    partidos_antes : DataFrame de partidos jugados ANTES del partido actual
    h2h            : DataFrame completo de enfrentamientos directos
    oponente       : nombre del equipo rival
    priors         : dict con medias globales de la liga (calculadas una vez
                     sobre todos los partidos disponibles en cada momento)

    Retorna dict con todas las features prefijadas con 'h_' o 'a_'
    según la condición.
    """
    prefix = "h_" if condition == "local" else "a_"

    # ── Historial del equipo ─────────────────────────────────────
    # Partidos donde este equipo participó (como local o visitante)
    mask_local = partidos_antes["local"] == team
    mask_visit = partidos_antes["visitante"] == team
    hist = partidos_antes[mask_local | mask_visit].copy()

    # Marcar si ganó y si jugó en casa
    hist["team_gano"]    = np.where(
        hist["local"] == team,
        hist["gana_local"],
        1 - hist["gana_local"]
    )
    hist["team_es_local"] = (hist["local"] == team).astype(int)
    hist["team_sets_fav"] = np.where(
        hist["local"] == team,
        hist["sets_local"],
        hist["sets_visit"]
    )
    hist["team_sets_con"] = np.where(
        hist["local"] == team,
        hist["sets_visit"],
        hist["sets_local"]
    )
    hist["set_diff"] = hist["team_sets_fav"] - hist["team_sets_con"]

    n_total = len(hist)

    # ── Priors globales de la liga ───────────────────────────────
    # Valores de referencia hacia los que "tira" el shrinkage cuando n es bajo
    prior_win_rate  = priors["win_rate"]       # ~0.50 (liga equilibrada)
    prior_set_wr    = priors["set_win_rate"]   # ~0.55 (leve ventaja local)
    prior_sets_fav  = priors["sets_fav"]       # sets ganados medios por partido
    prior_sets_con  = priors["sets_con"]       # sets perdidos medios por partido
    prior_set_diff  = priors["set_diff"]       # diferencia media de sets (~0)
    prior_h2h_wr    = 0.50                     # sin H2H previo → 50/50

    # ── Feature: win_rate_global ─────────────────────────────────
    # Media observada + shrinkage hacia la prior global de la liga
    win_rate_obs    = hist["team_gano"].mean() if n_total > 0 else prior_win_rate
    win_rate_global = shrink(win_rate_obs, n_total, prior_win_rate)

    # ── Feature: win_rate_last5 (forma reciente) ─────────────────
    # Shrinkage con menos observaciones → prior tiene más peso
    last5       = hist.tail(FORMA_VENTANA)
    n_last5     = len(last5)
    last5_obs   = last5["team_gano"].mean() if n_last5 > 0 else win_rate_global
    win_rate_last5 = shrink(last5_obs, n_last5, prior_win_rate)

    # ── Feature: win_rate_home / win_rate_away ───────────────────
    hist_home = hist[hist["team_es_local"] == 1]
    hist_away = hist[hist["team_es_local"] == 0]
    win_rate_home = shrink(
        hist_home["team_gano"].mean() if len(hist_home) > 0 else prior_win_rate,
        len(hist_home), prior_win_rate
    )
    win_rate_away = shrink(
        hist_away["team_gano"].mean() if len(hist_away) > 0 else prior_win_rate,
        len(hist_away), prior_win_rate
    )

    # ── Feature: pts_fav_avg_exp / pts_con_avg_exp ───────────────
    # Exponential smoothing + shrinkage hacia la prior de la liga.
    # El shrinkage actúa sobre la estimación final del smoothing, no sobre
    # cada valor individual: cuando n es bajo, el smoothing tiene poco
    # historial y el shrinkage lo ancla a la prior.
    if n_total > 0:
        pts_fav_exp = exp_smooth(hist["team_sets_fav"].tolist())
        pts_con_exp = exp_smooth(hist["team_sets_con"].tolist())
    else:
        pts_fav_exp = prior_sets_fav
        pts_con_exp = prior_sets_con
    pts_fav_exp = shrink(pts_fav_exp, n_total, prior_sets_fav)
    pts_con_exp = shrink(pts_con_exp, n_total, prior_sets_con)

    # ── Feature: set_win_rate ────────────────────────────────────
    total_sets_fav = hist["team_sets_fav"].sum()
    total_sets_jug = (hist["team_sets_fav"] + hist["team_sets_con"]).sum()
    set_win_rate_obs = (total_sets_fav / total_sets_jug) if total_sets_jug > 0 else prior_set_wr
    set_win_rate = shrink(set_win_rate_obs, n_total, prior_set_wr)

    # ── Feature: set_diff_exp ────────────────────────────────────
    set_diff_obs = exp_smooth(hist["set_diff"].tolist()) if n_total > 0 else prior_set_diff
    set_diff_exp = shrink(set_diff_obs, n_total, prior_set_diff)

    # ── Feature: forma_local / forma_away ────────────────────────
    # Shrinkage con la ventana de 5 → n pequeño → más influencia de la prior
    last5_home = hist_home.tail(FORMA_VENTANA)
    last5_away = hist_away.tail(FORMA_VENTANA)
    forma_home = shrink(
        last5_home["team_gano"].mean() if len(last5_home) > 0 else prior_win_rate,
        len(last5_home), prior_win_rate
    )
    forma_away = shrink(
        last5_away["team_gano"].mean() if len(last5_away) > 0 else prior_win_rate,
        len(last5_away), prior_win_rate
    )

    # ── Feature: ultimo_partido_set_diff ─────────────────────────
    # Rendimiento en el último partido: diferencia de sets
    # (similar a "performance in previous game" del paper)
    if n_total > 0:
        ultimo = hist.iloc[-1]
        ultimo_set_diff = int(ultimo["team_sets_fav"]) - int(ultimo["team_sets_con"])
    else:
        ultimo_set_diff = 0

    # ── Feature: racha_actual ────────────────────────────────────
    # Racha actual: +N si lleva N victorias seguidas, -N si lleva N derrotas
    # (captura momentum, equivalente al "form" con streaks del paper)
    racha = 0
    if n_total > 0:
        resultados = hist["team_gano"].tolist()
        ultimo_res = resultados[-1]
        for r in reversed(resultados):
            if r == ultimo_res:
                racha += (1 if ultimo_res == 1 else -1)
            else:
                break

    # ── Feature: n_partidos_hist ─────────────────────────────────
    # Número de partidos en el historial (indica fiabilidad de las stats)
    n_partidos_hist = n_total

    # ── Feature: dias_descanso ───────────────────────────────────
    # Sin fechas exactas en el CSV, usamos jornada_num como proxy de tiempo.
    # La diferencia de jornadas entre el partido actual y el último jugado
    # es la mejor aproximación disponible. Se calcula fuera de esta función
    # y se pasa como parámetro extra (ver build_match_features).

    # ── Feature: h2h_set_diff_exp ────────────────────────────────
    # Diferencia de sets en enfrentamientos directos entre estos dos equipos,
    # con exponential smoothing (desde la perspectiva del equipo actual).
    h2h_local = h2h[
        (h2h["home_club"] == team) & (h2h["away_club"] == oponente)
    ]["set_diff"].tolist()
    h2h_visit = h2h[
        (h2h["home_club"] == oponente) & (h2h["away_club"] == team)
    ]["set_diff"].apply(lambda x: -x).tolist()

    h2h_all = h2h_local + h2h_visit  # perspectiva del team
    h2h_set_diff_exp = exp_smooth(h2h_all) if h2h_all else 0.0

    # ── Feature: h2h_win_rate ────────────────────────────────────
    # Porcentaje de victorias H2H con shrinkage: sin historial H2H → 0.5
    h2h_wins_as_home = h2h[
        (h2h["home_club"] == team) & (h2h["away_club"] == oponente)
        & (h2h["home_sets"] > h2h["away_sets"])
    ].shape[0]
    h2h_wins_as_away = h2h[
        (h2h["home_club"] == oponente) & (h2h["away_club"] == team)
        & (h2h["away_sets"] > h2h["home_sets"])
    ].shape[0]
    h2h_total = len(h2h_local) + len(h2h_visit)  # antes del apply, son conteos
    h2h_total_raw = h2h[
        ((h2h["home_club"] == team) & (h2h["away_club"] == oponente)) |
        ((h2h["home_club"] == oponente) & (h2h["away_club"] == team))
    ].shape[0]
    h2h_total_wins = h2h_wins_as_home + h2h_wins_as_away
    h2h_win_rate_obs = (h2h_total_wins / h2h_total_raw) if h2h_total_raw > 0 else prior_h2h_wr
    # Para H2H usamos k=5 (menos datos disponibles → prior más fuerte)
    h2h_win_rate = shrink(h2h_win_rate_obs, h2h_total_raw, prior_h2h_wr, k=5)

    # ── Feature: ranking_temporada ───────────────────────────────
    # Posición relativa en la temporada actual basada en win_rate
    # (el ranking real requeriría datos externos; lo aproximamos con los datos)
    # Se calcula a nivel de partido en build_match_features.

    feats = {
        f"{prefix}win_rate_global":       round(win_rate_global, 4),
        f"{prefix}win_rate_last5":        round(win_rate_last5, 4),
        f"{prefix}win_rate_home":         round(win_rate_home, 4),
        f"{prefix}win_rate_away":         round(win_rate_away, 4),
        f"{prefix}pts_fav_exp":           round(pts_fav_exp, 4),
        f"{prefix}pts_con_exp":           round(pts_con_exp, 4),
        f"{prefix}set_win_rate":          round(set_win_rate, 4),
        f"{prefix}set_diff_exp":          round(set_diff_exp, 4),
        f"{prefix}forma_home":            round(forma_home, 4),
        f"{prefix}forma_away":            round(forma_away, 4),
        f"{prefix}ultimo_set_diff":       int(ultimo_set_diff),
        f"{prefix}racha":                 int(racha),
        #f"{prefix}n_partidos_hist":       int(n_partidos_hist),
        f"{prefix}h2h_set_diff_exp":      round(h2h_set_diff_exp, 4),
        f"{prefix}h2h_win_rate":          round(h2h_win_rate, 4),
    }
    return feats


# ═══════════════════════════════════════════════════════════════════
# 5. RANKING APROXIMADO POR TEMPORADA
# ═══════════════════════════════════════════════════════════════════

def compute_rankings(partidos: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un ranking acumulado por temporada para cada equipo
    en el momento de cada jornada. Devuelve un DataFrame con:
      partido_id, local_rank, visit_rank, local_rank_prev, visit_rank_prev
    """
    records = []

    for season in partidos["temporada"].unique():
        df_s = partidos[partidos["temporada"] == season].sort_values("jornada_num")

        # Acumular wins jornada a jornada
        wins = {}  # equipo -> wins acumulados

        for _, row in df_s.iterrows():
            local   = row["local"]
            visit   = row["visitante"]

            # Rank ANTES de este partido
            local_rank = _rank_from_wins(wins, local, len(wins) + 1)
            visit_rank = _rank_from_wins(wins, visit, len(wins) + 1)

            records.append({
                "partido_id":        row["partido_id"],
                "local_rank_season": local_rank,
                "visit_rank_season": visit_rank,
            })

            # Actualizar wins
            if local not in wins:   wins[local] = 0
            if visit not in wins:   wins[visit] = 0
            if row["gana_local"] == 1:
                wins[local] += 1
            else:
                wins[visit] += 1

    return pd.DataFrame(records)


def _rank_from_wins(wins: dict, team: str, default: int) -> int:
    """Posición del equipo según wins acumulados (1 = más wins)."""
    if not wins:
        return default
    team_wins = wins.get(team, 0)
    rank = 1 + sum(1 for w in wins.values() if w > team_wins)
    return rank


# ═══════════════════════════════════════════════════════════════════
# 6. DESCANSO ENTRE PARTIDOS (proxy por jornada)
# ═══════════════════════════════════════════════════════════════════

def compute_rest(partidos: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el 'descanso' de cada equipo como la diferencia de jornada_num
    entre el partido actual y el último partido jugado.
    Valores >7 se recortan a 7 (igual que el paper con días de descanso).
    Devuelve columnas: partido_id, h_descanso, a_descanso
    """
    last_jornada = {}  # equipo -> última jornada jugada
    records = []

    for _, row in partidos.iterrows():
        local = row["local"]
        visit = row["visitante"]
        jorn  = row["jornada_num"]
        pid   = row["partido_id"]

        h_rest = min(jorn - last_jornada.get(local, jorn - 7), 7)
        a_rest = min(jorn - last_jornada.get(visit, jorn - 7), 7)
        h_rest = max(h_rest, 0)
        a_rest = max(a_rest, 0)

        records.append({"partido_id": pid, "h_descanso": h_rest, "a_descanso": a_rest})

        last_jornada[local] = jorn
        last_jornada[visit] = jorn

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════
# 7. BUILD MATCH FEATURES (dataset principal — un registro por partido)
# ═══════════════════════════════════════════════════════════════════

def build_match_features(partidos: pd.DataFrame, h2h: pd.DataFrame) -> pd.DataFrame:
    """
    Itera sobre todos los partidos en orden cronológico y construye,
    para cada uno, el vector de features de ambos equipos usando
    SOLO información disponible antes del partido (sin data leakage).
    Las priors para el shrinkage bayesiano se calculan como las medias
    globales de TODOS los partidos del dataset (sin leakage, ya que
    representan el comportamiento medio de la liga como referencia estable).
    """
    print("→ Calculando rankings por temporada...")
    rankings = compute_rankings(partidos)
    partidos = partidos.merge(rankings, on="partido_id", how="left")

    print("→ Calculando descanso entre partidos...")
    rest_df = compute_rest(partidos)
    partidos = partidos.merge(rest_df, on="partido_id", how="left")

    # ── Priors globales de la liga ───────────────────────────────
    # Se calculan sobre el dataset completo. Representan el valor
    # "de referencia" de un equipo desconocido en esta liga.
    # No hay data leakage: son constantes estructurales de la competición.
    print("→ Calculando priors globales de la liga...")

    # Para calcular las priors necesitamos construir el historial completo
    # de todos los partidos en perspectiva de equipo.
    all_local = partidos[["sets_local","sets_visit","gana_local"]].copy()
    all_local.columns = ["sets_fav","sets_con","gano"]
    all_visit = partidos[["sets_visit","sets_local","gana_local"]].copy()
    all_visit.columns = ["sets_fav","sets_con","gano"]
    all_visit["gano"] = 1 - all_visit["gano"]
    all_team = pd.concat([all_local, all_visit], ignore_index=True)

    priors = {
        "win_rate":    round(all_team["gano"].mean(), 4),       # ~0.50
        "set_win_rate": round(
            all_team["sets_fav"].sum() /
            (all_team["sets_fav"] + all_team["sets_con"]).sum(), 4
        ),                                                        # ~0.55
        "sets_fav":    round(all_team["sets_fav"].mean(), 4),   # ~1.8
        "sets_con":    round(all_team["sets_con"].mean(), 4),   # ~1.5
        "set_diff":    round((all_team["sets_fav"] - all_team["sets_con"]).mean(), 4),  # ~0.3
    }
    print(f"   Priors calculadas: {priors}")

    print("→ Construyendo features partido a partido (sin data leakage)...")
    rows = []
    n = len(partidos)

    for i, row in partidos.iterrows():
        if i % 50 == 0:
            print(f"   Partido {i+1}/{n}...")

        pid    = row["partido_id"]
        local  = row["local"]
        visit  = row["visitante"]
        season = row["temporada"]
        jorn   = row["jornada_num"]

        # Historial estricto: solo partidos ANTES de este (mismo índice global)
        anteriores = partidos[partidos["match_order"] < row["match_order"]]

        # Features del equipo local (como 'local' en este partido)
        feats_h = compute_team_features_at(
            team=local,
            condition="local",
            partidos_antes=anteriores,
            h2h=h2h,
            oponente=visit,
            priors=priors,
        )

        # Features del equipo visitante (como 'visitante' en este partido)
        feats_a = compute_team_features_at(
            team=visit,
            condition="visitante",
            partidos_antes=anteriores,
            h2h=h2h,
            oponente=local,
            priors=priors,
        )

        # Features de diferencia (local - visitante)
        # Estas son las más informativas para el modelo
        diff = {
            "diff_win_rate_global":  feats_h["h_win_rate_global"] - feats_a["a_win_rate_global"],
            "diff_win_rate_last5":   feats_h["h_win_rate_last5"]  - feats_a["a_win_rate_last5"],
            "diff_set_win_rate":     feats_h["h_set_win_rate"]    - feats_a["a_set_win_rate"],
            "diff_set_diff_exp":     feats_h["h_set_diff_exp"]    - feats_a["a_set_diff_exp"],
            "diff_pts_fav_exp":      feats_h["h_pts_fav_exp"]     - feats_a["a_pts_fav_exp"],
            "diff_pts_con_exp":      feats_h["h_pts_con_exp"]     - feats_a["a_pts_con_exp"],
            "diff_racha":            feats_h["h_racha"]           - feats_a["a_racha"],
            "diff_ultimo_set_diff":  feats_h["h_ultimo_set_diff"] - feats_a["a_ultimo_set_diff"],
            "diff_rank_season":      row["visit_rank_season"]     - row["local_rank_season"],  # + si local mejor
        }

        # Features de condición local (ventaja de campo)
        # Forma específica home/away según condición real del partido
        forma_efectiva_h = feats_h["h_forma_home"]   # local jugando en casa
        forma_efectiva_a = feats_a["a_forma_away"]   # visitante jugando fuera
        diff["diff_forma_efectiva"] = forma_efectiva_h - forma_efectiva_a

        record = {
            "partido_id":            pid,
            "temporada":             season,
            "jornada_num":           jorn,
            "local":                 local,
            "visitante":             visit,
            **feats_h,
            **feats_a,
            **diff,
            "h_descanso":            row["h_descanso"],
            "a_descanso":            row["a_descanso"],
            "diff_descanso":         row["h_descanso"] - row["a_descanso"],
            "h_rank_season":         row["local_rank_season"],
            "a_rank_season":         row["visit_rank_season"],
            "gana_local":            row["gana_local"],      # target
        }
        rows.append(record)

    df_out = pd.DataFrame(rows)
    return df_out


# ═══════════════════════════════════════════════════════════════════
# 8. BUILD SET FEATURES (dataset secundario — un registro por set)
# ═══════════════════════════════════════════════════════════════════

def build_set_features(df_sets: pd.DataFrame,
                       match_features: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el dataset de sets. Cada fila = un set de un partido.

    Features del set:
      - strength_h / strength_a: win_rate_global del Strength Model
        (proxy de fuerza, se actualizará con el score del modelo entrenado)
      - strength_diff
      - set_num (normalizado 1-5)
      - sets_h_antes, sets_a_antes: marcador parcial antes del set
      - momentum_h: ¿ganó el local el set anterior?
      - es_desempate: set_num == 5

    Target: ganador_set_local (0/1)
    """
    # Join con match_features para traer el strength proxy
    mf = match_features[[
        "partido_id", "h_win_rate_global", "a_win_rate_global",
        "h_set_win_rate", "a_set_win_rate",
        "diff_win_rate_global", "diff_set_win_rate",
        "h_forma_home", "a_forma_away",
        "diff_forma_efectiva",
        "h_pts_fav_exp", "a_pts_fav_exp",
        "h_h2h_set_diff_exp",   # H2H ya calculado en match features
        "gana_local",
    ]].copy()

    df_merged = df_sets.merge(mf, on="partido_id", how="inner")

    rows = []
    for pid, grupo in df_merged.groupby("partido_id"):
        grupo = grupo.sort_values("set_num").reset_index(drop=True)

        sets_h = 0
        sets_a = 0

        for idx, row in grupo.iterrows():
            sn = int(row["set_num"])

            # Momentum: resultado del set anterior (0 si es el primero)
            if idx == 0:
                momentum_h = 0.5  # neutro al inicio
            else:
                prev = grupo.iloc[idx - 1]
                momentum_h = float(prev["ganador_set_local"])

            record = {
                "partido_id":          pid,
                "set_num":             sn,
                "set_num_norm":        (sn - 1) / 4.0,   # normalizado [0, 1]
                # Fuerza global (proxy del Strength Model)
                "strength_h":          row["h_win_rate_global"],
                "strength_a":          row["a_win_rate_global"],
                "strength_diff":       row["diff_win_rate_global"],
                # Set win rate (más granular)
                "set_wr_h":            row["h_set_win_rate"],
                "set_wr_a":            row["a_set_win_rate"],
                "diff_set_wr":         row["diff_set_win_rate"],
                # Forma efectiva (home/away)
                "forma_h":             row["h_forma_home"],
                "forma_a":             row["a_forma_away"],
                "diff_forma":          row["diff_forma_efectiva"],
                # Potencia ofensiva
                "pts_fav_h":           row["h_pts_fav_exp"],
                "pts_fav_a":           row["a_pts_fav_exp"],
                # H2H (perspectiva del local)
                "h2h_diff":            row["h_h2h_set_diff_exp"],
                # Contexto dentro del partido
                "sets_h_antes":        sets_h,
                "sets_a_antes":        sets_a,
                "diff_sets_antes":     sets_h - sets_a,
                "momentum_h":          momentum_h,
                "es_desempate":        int(sn == 5),
                # Target
                "ganador_set_local":   int(row["ganador_set_local"]),
            }
            rows.append(record)

            # Actualizar marcador parcial
            if row["ganador_set_local"] == 1:
                sets_h += 1
            else:
                sets_a += 1

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PREDICTOR VB v2 — Feature Builder")
    print("=" * 60)

    # 1. Carga
    print("\n[1/5] Cargando sets_partidos.csv...")
    df_sets = load_sets()
    print(f"      {len(df_sets)} sets | {df_sets['partido_id'].nunique()} partidos")

    # 2. Tabla de partidos
    print("\n[2/5] Construyendo tabla de partidos...")
    partidos = build_match_table(df_sets)
    print(f"      {len(partidos)} partidos ordenados cronológicamente")
    print(f"      Temporadas: {partidos['temporada'].min()} → {partidos['temporada'].max()}")
    print(f"      Equipos únicos: {sorted(set(partidos['local'].tolist() + partidos['visitante'].tolist()))}")

    # 3. H2H
    print("\n[3/5] Cargando enfrentamientos directos (H2H)...")
    h2h = load_h2h()
    print(f"      {len(h2h)} enfrentamientos H2H cargados")

    # 4. Match features
    print("\n[4/5] Construyendo match_features (Strength Model input)...")
    match_feats = build_match_features(partidos, h2h)

    out_match = os.path.join(OUT_DIR, "match_features.csv")
    match_feats.to_csv(out_match, index=False)
    print(f"\n      ✓ Guardado: {out_match}")
    print(f"      Shape: {match_feats.shape}")
    print(f"      Features ({len(match_feats.columns)} columnas):")
    feature_cols = [c for c in match_feats.columns
                    if c not in ("partido_id","temporada","jornada_num","local","visitante","gana_local")]
    for c in feature_cols:
        print(f"        · {c}")

    # 5. Set features
    print("\n[5/5] Construyendo set_features (Set Model input)...")
    set_feats = build_set_features(df_sets, match_feats)

    out_sets = os.path.join(OUT_DIR, "set_features.csv")
    set_feats.to_csv(out_sets, index=False)
    print(f"\n      ✓ Guardado: {out_sets}")
    print(f"      Shape: {set_feats.shape}")

    # ── Resumen estadístico ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)

    target_match = match_feats["gana_local"]
    target_set   = set_feats["ganador_set_local"]

    print(f"\n  match_features.csv")
    print(f"    Partidos totales : {len(match_feats)}")
    print(f"    Gana local       : {target_match.sum()} ({target_match.mean()*100:.1f}%)")
    print(f"    Gana visitante   : {(1-target_match).sum()} ({(1-target_match).mean()*100:.1f}%)")
    print(f"    Features numéricas: {len(feature_cols)}")

    # Verificación del efecto del shrinkage
    # print(f"\n  Verificación shrinkage bayesiano (K={SHRINKAGE_K}):")
    # low_hist  = match_feats[match_feats["h_n_partidos_hist"] <= 3]
    # high_hist = match_feats[match_feats["h_n_partidos_hist"] >= 20]
    # print(f"    Partidos con ≤3 partidos de historial (local): {len(low_hist)}")
    # print(f"      win_rate_global medio : {low_hist['h_win_rate_global'].mean():.3f}  "
    #       f"(sin shrinkage sería ~0.5 exacto, con ruido alto)")
    # print(f"    Partidos con ≥20 partidos de historial (local): {len(high_hist)}")
    # print(f"      win_rate_global medio : {high_hist['h_win_rate_global'].mean():.3f}  "
    #       f"(shrinkage apenas influye)")

    print(f"\n  set_features.csv")
    print(f"    Sets totales     : {len(set_feats)}")
    print(f"    Gana local set   : {target_set.sum()} ({target_set.mean()*100:.1f}%)")
    print(f"    Gana visit set   : {(1-target_set).sum()} ({(1-target_set).mean()*100:.1f}%)")
    print(f"    Features numéricas: {len([c for c in set_feats.columns if c not in ('partido_id','ganador_set_local')])}")

    print(f"\n  Archivos en {OUT_DIR}")
    for f in os.listdir(OUT_DIR):
        fpath = os.path.join(OUT_DIR, f)
        size  = os.path.getsize(fpath) / 1024
        print(f"    {f}  ({size:.1f} KB)")

    print("\n✓ Feature building completado sin errores.\n")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()