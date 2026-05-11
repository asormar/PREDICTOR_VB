"""
feature_engineering.py
-----------------------
Builds a match-level feature matrix from the volleyball dataset.

Features per match:
  - Recent form of each team (last N matches in same season)
  - Head-to-head historical balance (sets won/lost)
  - Season team stats (attack, serve, block, reception efficiencies)
  - Top player stats aggregated per team (points/set, aces/set, etc.)
  - Match context (jornada number, fase encoded)

Target variables:
  - match_result: 0=3-0, 1=3-1, 2=3-2, 3=0-3, 4=1-3, 5=2-3  (6-class)
  - local_wins: 1 if local team wins the match, 0 otherwise  (binary)
"""

import os
import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

RAW = Path(__file__).parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# 1. Team name normalisation
# ---------------------------------------------------------------------------

CANONICAL = {
    # Trento
    "diatec trentino": "Trento",
    "itas trentino": "Trento",
    "trento": "Trento",
    # Perugia
    "sir safety conad perugia": "Perugia",
    "sir safety susa perugia": "Perugia",
    "sir susa vim perugia": "Perugia",
    "perugia": "Perugia",
    # Modena
    "azimut modena": "Modena",
    "azimut leo shoes modena": "Modena",
    "leo shoes perkinelmer modena": "Modena",
    "modena": "Modena",
    # Lube / Civitanova
    "lube civitanova": "Lube",
    "cucine lube civitanova": "Lube",
    "lube": "Lube",
    # Milano
    "allianz milano": "Milano",
    "powervolley milano": "Milano",
    "milano": "Milano",
    # Monza
    "gi group monza": "Monza",
    "mint vero volley monza": "Monza",
    "vero volley monza": "Monza",
    "monza": "Monza",
    # Padova
    "kioene padova": "Padova",
    "pallavolo padova": "Padova",
    "padova": "Padova",
    # Piacenza
    "piacenza": "Piacenza",
    "gas sales bluenergy piacenza": "Piacenza",
    # Verona
    "verona volley": "Verona",
    "verona": "Verona",
    # Taranto
    "gioiella prisma taranto": "Taranto",
    "prisma taranto": "Taranto",
    "taranto": "Taranto",
    # Grottazzolina
    "videx grottazzolina": "Grottazzolina",
    "videx yuasa grottazzolina": "Grottazzolina",
    "grottazzolina": "Grottazzolina",
    # Cisterna
    "top volley cisterna": "Cisterna",
    "cisterna": "Cisterna",
    # Ravenna
    "consar rcm ravenna": "Ravenna",
    "ravenna": "Ravenna",
    # Cuneo
    "bam acqua s. bernardo cuneo": "Cuneo",
    "banca alpi marittime acqua s.bernardo cuneo": "Cuneo",
    "puliservice acqua s.bernardo cuneo": "Cuneo",
    "cuneo": "Cuneo",
    # Siena
    "emma villas aubay siena": "Siena",
    "emma villas siena": "Siena",
    "siena": "Siena",
    # Prata
    "tinet gori wines prata di pordenone": "Prata",
    "prata di pordenone": "Prata",
    # Brescia
    "consoli mcdonald's brescia": "Brescia",
    "brescia": "Brescia",
    # Others (keep as-is simplified)
    "pineto": "Pineto",
    "porto viro": "Porto Viro",
}


def normalise_team(name: str) -> str:
    """Lower-strip-lookup, fallback to title-cased original."""
    key = re.sub(r"\s+", " ", str(name).strip()).lower()
    # enfrentamientos files duplicate the name: "MilanoMilano"
    key = re.sub(r"^(.+)\1$", r"\1", key)
    return CANONICAL.get(key, name.strip())


# ---------------------------------------------------------------------------
# 2. Load & clean sets_partidos
# ---------------------------------------------------------------------------

def load_sets() -> pd.DataFrame:
    df = pd.read_csv("DB/sets_partidos.csv")
    df["equipo_local"] = df["equipo_local"].apply(normalise_team)
    df["equipo_visitante"] = df["equipo_visitante"].apply(normalise_team)
    # Season as integer start year for ordering
    df["season_year"] = df["temporada"].str[:4].astype(int)
    return df


def build_match_df(sets: pd.DataFrame) -> pd.DataFrame:
    """One row per match with basic info."""
    matches = sets.drop_duplicates("partido_id").copy()
    matches = matches[[
        "partido_id", "temporada", "season_year", "jornada", "fase",
        "equipo_local", "equipo_visitante", "resultado_final"
    ]].reset_index(drop=True)

    # Encode match result (6 classes)
    result_map = {"3-0": 0, "3-1": 1, "3-2": 2, "0-3": 3, "1-3": 4, "2-3": 5}
    matches["match_result"] = matches["resultado_final"].map(result_map)
    matches["local_wins"] = (matches["match_result"] < 3).astype(int)

    # Jornada as numeric (extract first number)
    matches["jornada_num"] = (
        matches["jornada"].str.extract(r"(\d+)")[0].astype(float)
    )

    # Fase encoding
    fase_map = {"1st half": 0, "2nd half": 1, "Playoffs": 2, "Playoff": 2,
                "Final": 3, "Semifinal": 2, "Quarterfinal": 1}
    matches["fase_enc"] = matches["fase"].map(
        lambda x: next((v for k, v in fase_map.items() if k.lower() in str(x).lower()), 0)
    )

    return matches


# ---------------------------------------------------------------------------
# 3. Recent form features
# ---------------------------------------------------------------------------

def _team_form(matches: pd.DataFrame, team: str, before_id: str,
               same_season: str, n: int = 5) -> dict:
    """Compute last-n-match stats for `team` before `before_id`."""
    season_mask = matches["temporada"] == same_season
    idx = matches.index[matches["partido_id"] == before_id]
    if len(idx) == 0:
        return {}
    i = idx[0]

    prev = matches[(matches.index < i) & season_mask].copy()
    as_local = prev[prev["equipo_local"] == team].copy()
    as_visit = prev[prev["equipo_visitante"] == team].copy()

    as_local["won"] = (as_local["match_result"] < 3).astype(int)
    as_local["sets_favor"] = as_local["resultado_final"].str.split("-").str[0].astype(int)
    as_local["sets_contra"] = as_local["resultado_final"].str.split("-").str[1].astype(int)

    as_visit["won"] = (as_visit["match_result"] >= 3).astype(int)
    as_visit["sets_favor"] = as_visit["resultado_final"].str.split("-").str[1].astype(int)
    as_visit["sets_contra"] = as_visit["resultado_final"].str.split("-").str[0].astype(int)

    combined = pd.concat([as_local, as_visit]).sort_index().tail(n)

    if len(combined) == 0:
        return {"win_rate": 0.5, "avg_sets_favor": 1.5, "avg_sets_contra": 1.5,
                "n_games": 0}

    return {
        "win_rate": combined["won"].mean(),
        "avg_sets_favor": combined["sets_favor"].mean(),
        "avg_sets_contra": combined["sets_contra"].mean(),
        "n_games": len(combined),
    }


def add_form_features(matches: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Add rolling form features for both teams."""
    records = []
    for _, row in matches.iterrows():
        local_form = _team_form(matches, row["equipo_local"], row["partido_id"],
                                row["temporada"], n)
        visit_form = _team_form(matches, row["equipo_visitante"], row["partido_id"],
                                row["temporada"], n)

        records.append({
            "partido_id": row["partido_id"],
            "local_win_rate": local_form.get("win_rate", 0.5),
            "local_avg_sets_favor": local_form.get("avg_sets_favor", 1.5),
            "local_avg_sets_contra": local_form.get("avg_sets_contra", 1.5),
            "local_n_games": local_form.get("n_games", 0),
            "visit_win_rate": visit_form.get("win_rate", 0.5),
            "visit_avg_sets_favor": visit_form.get("avg_sets_favor", 1.5),
            "visit_avg_sets_contra": visit_form.get("avg_sets_contra", 1.5),
            "visit_n_games": visit_form.get("n_games", 0),
        })

    form_df = pd.DataFrame(records)
    return matches.merge(form_df, on="partido_id")


# ---------------------------------------------------------------------------
# 4. Head-to-head features
# ---------------------------------------------------------------------------

def load_h2h() -> pd.DataFrame:
    """Load and concatenate all head-to-head files."""
    frames = []
    h2h_dir = RAW / "enfrentamientos_directos"
    for f in sorted(h2h_dir.glob("enfrentamientos_directos_*.csv")):
        df = pd.read_csv(f)
        frames.append(df)
    h2h = pd.concat(frames, ignore_index=True)
    h2h["home_club"] = h2h["home_club"].apply(normalise_team)
    h2h["away_club"] = h2h["away_club"].apply(normalise_team)
    return h2h


def _h2h_stats(h2h: pd.DataFrame, local: str, visitante: str,
               before_season_year: int) -> dict:
    """Historical sets balance between two teams before a given season."""
    mask = (
        ((h2h["home_club"] == local) & (h2h["away_club"] == visitante)) |
        ((h2h["home_club"] == visitante) & (h2h["away_club"] == local))
    )
    season_year = h2h["season"].str[:4].astype(int)
    relevant = h2h[mask & (season_year < before_season_year)]

    if len(relevant) == 0:
        return {"h2h_local_sets": 0, "h2h_visit_sets": 0, "h2h_matches": 0,
                "h2h_local_winrate": 0.5}

    local_sets = 0
    visit_sets = 0
    local_wins = 0
    for _, r in relevant.iterrows():
        if r["home_club"] == local:
            local_sets += r["home_sets"]
            visit_sets += r["away_sets"]
            if r["home_sets"] > r["away_sets"]:
                local_wins += 1
        else:
            local_sets += r["away_sets"]
            visit_sets += r["home_sets"]
            if r["away_sets"] > r["home_sets"]:
                local_wins += 1

    n = len(relevant)
    return {
        "h2h_local_sets": local_sets,
        "h2h_visit_sets": visit_sets,
        "h2h_matches": n,
        "h2h_local_winrate": local_wins / n,
    }


def add_h2h_features(matches: pd.DataFrame, h2h: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in matches.iterrows():
        stats = _h2h_stats(h2h, row["equipo_local"], row["equipo_visitante"],
                           row["season_year"])
        stats["partido_id"] = row["partido_id"]
        records.append(stats)
    return matches.merge(pd.DataFrame(records), on="partido_id")


# ---------------------------------------------------------------------------
# 5. Season team stats
# ---------------------------------------------------------------------------

def load_team_stats() -> pd.DataFrame:
    path = list(RAW.glob("Comparacion_equipos_10_a*.csv"))[0]
    df = pd.read_csv(path)
    # Flatten multi-level column names
    df.columns = ["_".join(str(c) for c in col).strip("_") if isinstance(col, tuple)
                  else str(col) for col in df.columns]
    df.rename(columns={"Club_Club": "team", "Temporada": "season_year"}, inplace=True)
    df["team"] = df["team"].apply(normalise_team)

    keep = ["team", "season_year",
            "SERVE_Ace per Set", "SERVE_Effic.",
            "RECEPTION_Exc. %", "RECEPTION_Effic.",
            "ATTACK_Exc. %", "ATTACK_Effic.",
            "BLOCK_Points per Set"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()
    df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce")
    df.dropna(subset=["season_year"], inplace=True)
    df["season_year"] = df["season_year"].astype(int)
    return df


def add_team_stats(matches: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    stat_cols = [c for c in team_stats.columns if c not in ("team", "season_year")]

    def _merge_side(side_col, prefix):
        side = team_stats.rename(columns={c: f"{prefix}_{c}" for c in stat_cols})
        side = side.rename(columns={"team": side_col})
        merged = matches.merge(
            side, left_on=[side_col, "season_year"],
            right_on=[side_col, "season_year"], how="left"
        )
        return merged

    matches = _merge_side("equipo_local", "local_stat")
    matches = _merge_side("equipo_visitante", "visit_stat")
    return matches


# ---------------------------------------------------------------------------
# 6. Player stats aggregated per team
# ---------------------------------------------------------------------------

def load_player_stats() -> pd.DataFrame:
    """Aggregate player stats per team per season."""
    files = {
        "points_per_set": "Points_Set_filtered.xlsx",
        "aces_per_set": "Ace_Set_filtered.xlsx",
        "attacks_per_set": "Won_Attacks_Set_filtered.xlsx",
        "blocks_per_set": "Won_Blocks_Set_filtered.xlsx",
        "receptions_per_set": "Excellent_Receptions_Set_filtered.xlsx",
    }
    stat_dir = RAW / "stats_jugadores_set"
    frames = []

    for stat_name, fname in files.items():
        fpath = stat_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_excel(fpath, header=0)
        # Drop rows that are fully None
        df.dropna(how="all", inplace=True)
        # Rename columns positionally: Player, Role, Team, Season, Matches, Sets, stat, stat_per_set
        cols = df.columns.tolist()
        # Find the per-set column (last numeric-like column)
        per_set_col = cols[-1] if pd.api.types.is_numeric_dtype(df[cols[-1]]) else cols[-2]
        df = df.rename(columns={cols[2]: "team", cols[3]: "season", per_set_col: stat_name})
        df = df[["team", "season", stat_name]].dropna()
        df["team"] = df["team"].apply(normalise_team)
        df["season_year"] = df["season"].astype(str).str[:4].str.extract(r"(\d{4})")[0]
        df["season_year"] = pd.to_numeric(df["season_year"], errors="coerce").astype("Int64")
        frames.append(df[["team", "season_year", stat_name]])

    if not frames:
        return pd.DataFrame()

    # Merge all stats
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["team", "season_year"], how="outer")

    # Aggregate to team-season level (mean of top players)
    result = result.groupby(["team", "season_year"]).mean(numeric_only=True).reset_index()
    return result


def add_player_stats(matches: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
    if player_stats.empty:
        return matches
    stat_cols = [c for c in player_stats.columns if c not in ("team", "season_year")]

    for side_col, prefix in [("equipo_local", "local_pl"), ("equipo_visitante", "visit_pl")]:
        side = player_stats.rename(columns={c: f"{prefix}_{c}" for c in stat_cols})
        matches = matches.merge(
            side, left_on=[side_col, "season_year"],
            right_on=["team", "season_year"], how="left"
        ).drop(columns=["team"], errors="ignore")

    return matches


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------

def build_features(form_n: int = 5) -> pd.DataFrame:
    """
    Full pipeline. Returns a DataFrame with one row per match,
    including all feature columns and target columns.
    """
    print("Loading sets data...")
    sets = load_sets()
    matches = build_match_df(sets)

    print("Computing form features...")
    matches = add_form_features(matches, n=form_n)

    print("Loading H2H data...")
    h2h = load_h2h()
    matches = add_h2h_features(matches, h2h)

    print("Loading team stats...")
    team_stats = load_team_stats()
    matches = add_team_stats(matches, team_stats)

    print("Loading player stats...")
    player_stats = load_player_stats()
    matches = add_player_stats(matches, player_stats)

    print(f"Final dataset: {matches.shape[0]} matches, {matches.shape[1]} columns")
    print(f"Missing values per column:\n{matches.isnull().sum()[matches.isnull().sum() > 0]}")

    return matches


if __name__ == "__main__":
    df = build_features()
    out = Path(__file__).parent.parent / "data" / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved to {out}")
    print(df[["equipo_local", "equipo_visitante", "temporada",
              "match_result", "local_wins"]].head(10))
