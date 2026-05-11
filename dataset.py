"""
dataset.py
----------
PyTorch Dataset and data preparation for the volleyball model.

Temporal split strategy:
  - Train:      seasons up to (and including) train_until_season
  - Validation: next season
  - Test:       most recent season(s)

This avoids data leakage: the model never sees future seasons during training.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# ---------------------------------------------------------------------------
# Feature columns definition
# ---------------------------------------------------------------------------

FORM_FEATURES = [
    "local_win_rate", "local_avg_sets_favor", "local_avg_sets_contra",
    "local_n_games", "visit_win_rate", "visit_avg_sets_favor",
    "visit_avg_sets_contra", "visit_n_games",
]

H2H_FEATURES = [
    "h2h_local_sets", "h2h_visit_sets", "h2h_matches", "h2h_local_winrate",
]

CONTEXT_FEATURES = [
    "jornada_num", "fase_enc",
]

TEAM_STAT_PREFIXES = ["local_stat", "visit_stat"]
TEAM_STAT_SUFFIXES = [
    "SERVE_Ace per Set", "SERVE_Effic.",
    "RECEPTION_Exc. %", "RECEPTION_Effic.",
    "ATTACK_Exc. %", "ATTACK_Effic.",
    "BLOCK_Points per Set",
]

PLAYER_STAT_NAMES = [
    "points_per_set", "aces_per_set", "attacks_per_set",
    "blocks_per_set", "receptions_per_set",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Determine which feature columns are actually present in df."""
    cols = []
    cols += [c for c in FORM_FEATURES if c in df.columns]
    cols += [c for c in H2H_FEATURES if c in df.columns]
    cols += [c for c in CONTEXT_FEATURES if c in df.columns]
    for prefix in TEAM_STAT_PREFIXES:
        for suffix in TEAM_STAT_SUFFIXES:
            col = f"{prefix}_{suffix}"
            if col in df.columns:
                cols.append(col)
    for prefix in ["local_pl", "visit_pl"]:
        for stat in PLAYER_STAT_NAMES:
            col = f"{prefix}_{stat}"
            if col in df.columns:
                cols.append(col)
    return cols


# ---------------------------------------------------------------------------
# Team indexer
# ---------------------------------------------------------------------------

class TeamIndexer:
    """Maps team names to integer indices (1-based, 0 = unknown)."""

    def __init__(self):
        self.team2idx: dict[str, int] = {}

    def fit(self, teams: list[str]) -> "TeamIndexer":
        unique = sorted(set(teams))
        self.team2idx = {t: i + 1 for i, t in enumerate(unique)}
        return self

    def transform(self, teams: list[str]) -> list[int]:
        return [self.team2idx.get(t, 0) for t in teams]

    @property
    def n_teams(self) -> int:
        return len(self.team2idx)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VolleyballDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        team_indexer: TeamIndexer,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.team_indexer = team_indexer

        # Fill missing numeric values with 0 (conservative – unknown stats)
        X = self.df[feature_cols].fillna(0).values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X).astype(np.float32)
        elif scaler is not None:
            self.scaler = scaler
            self.X = self.scaler.transform(X).astype(np.float32)
        else:
            self.scaler = None
            self.X = X

        self.local_idx = np.array(
            team_indexer.transform(self.df["equipo_local"].tolist()), dtype=np.int64
        )
        self.visit_idx = np.array(
            team_indexer.transform(self.df["equipo_visitante"].tolist()), dtype=np.int64
        )
        self.match_result = self.df["match_result"].values.astype(np.int64)
        self.local_wins = self.df["local_wins"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        return {
            "numeric": torch.tensor(self.X[idx]),
            "local_idx": torch.tensor(self.local_idx[idx]),
            "visit_idx": torch.tensor(self.visit_idx[idx]),
            "match_result": torch.tensor(self.match_result[idx]),
            "local_wins": torch.tensor(self.local_wins[idx]),
        }


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

SEASON_ORDER = [
    "2016/2017", "2017/2018", "2018/2019", "2019/2020", "2020/2021",
    "2021/2022", "2022/2023", "2023/2024", "2024/2025", "2025/2026",
]


def temporal_split(
    df: pd.DataFrame,
    val_season: str = "2024/2025",
    test_season: str = "2025/2026",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by season:
      train: everything before val_season
      val:   val_season
      test:  test_season
    """
    val_idx = SEASON_ORDER.index(val_season)
    train_seasons = set(SEASON_ORDER[:val_idx])
    val_seasons = {val_season}
    test_seasons = {test_season}

    train = df[df["temporada"].isin(train_seasons)]
    val = df[df["temporada"].isin(val_seasons)]
    test = df[df["temporada"].isin(test_seasons)]

    print(f"Train: {len(train)} matches ({sorted(train_seasons)})")
    print(f"Val:   {len(val)} matches ({sorted(val_seasons)})")
    print(f"Test:  {len(test)} matches ({sorted(test_seasons)})")
    return train, val, test


def make_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    val_season: str = "2024/2025",
    test_season: str = "2025/2026",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, TeamIndexer, StandardScaler, list[str]]:
    """
    Full pipeline: split → encode teams → scale → DataLoaders.
    Returns (train_loader, val_loader, test_loader, team_indexer, scaler, feature_cols)
    """
    train_df, val_df, test_df = temporal_split(df, val_season, test_season)
    feature_cols = get_feature_columns(df)

    # Fit team indexer on all teams (including test, for embedding coverage)
    all_teams = df["equipo_local"].tolist() + df["equipo_visitante"].tolist()
    team_indexer = TeamIndexer().fit(all_teams)

    # Train dataset (also fits the scaler)
    train_ds = VolleyballDataset(train_df, feature_cols, team_indexer, fit_scaler=True)
    val_ds = VolleyballDataset(val_df, feature_cols, team_indexer, scaler=train_ds.scaler)
    test_ds = VolleyballDataset(test_df, feature_cols, team_indexer, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    return train_loader, val_loader, test_loader, team_indexer, train_ds.scaler, feature_cols
