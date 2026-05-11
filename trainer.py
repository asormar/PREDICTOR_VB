"""
trainer.py
──────────
Entrena dos módulos independientes usando arquitectura Transformer:

  Módulo 1 — MatchTransformer
      Entrada : secuencia de los últimos 10 partidos de cada equipo
                (5 features derivadas del historial: set_win_rate,
                 pts_diff_avg, sets_largos_rate, pts_avg_favor, pts_avg_contra)
      Salida  : probabilidad sobre 6 resultados posibles
                (3-0, 3-1, 3-2, 0-3, 1-3, 2-3)

  Módulo 2 — SetTransformer
      Entrada : stats de los 5 jugadores clave de cada equipo (por temporada)
                5 jugadores × 5 stats = 25 features por equipo → 50 en total
      Salida  : probabilidad de que gane el set el equipo local (binario)

Equipos: Piacenza, Trento, Milano, Lube, Verona, Padova, Monza, Modena
Temporadas: 2021/2022 – 2025/2026

Uso:
    pip install torch pandas numpy scikit-learn openpyxl
    python trainer.py
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════

TOP8 = ["Piacenza", "Trento", "Milano", "Lube",
        "Verona", "Padova", "Monza", "Modena"]

SEASONS = ["2021/2022", "2022/2023", "2023/2024", "2024/2025", "2025/2026"]

SETS_CSV     = "DB/sets_partidos.csv"
PLAYER_FILES = {
    "pts_set": "DB/stats_jugadores_set/Points_Set_filtered.xlsx",
    "att_set": "DB/stats_jugadores_set/Won_Attacks_Set_filtered.xlsx",
    "ace_set": "DB/stats_jugadores_set/Ace_Set_filtered.xlsx",
    "rec_set": "DB/stats_jugadores_set/Excellent_Receptions_Set_filtered.xlsx",
    "blk_set": "DB/stats_jugadores_set/Won_Blocks_Set_filtered.xlsx",
}
PLAYER_STAT_COLS = {
    "pts_set": "Points / Set",
    "att_set": "Won Attacks / Set",
    "ace_set": "Ace / Set",
    "rec_set": "Excellent Receptions / Set",
    "blk_set": "Won Blocks / Set",
}

MODEL_DIR     = "model"
CONTEXT_LEN   = 10    # partidos recientes usados como contexto
TOP_PLAYERS   = 5     # jugadores clave por equipo
MATCH_RESULTS = ["3-0", "3-1", "3-2", "0-3", "1-3", "2-3"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════
# 0. NORMALIZACIÓN DE NOMBRES
# ═════════════════════════════════════════════════════════════════

CLUB_NAME_MAP = {
    "Gas Sales Bluenergy Piacenza":         "Piacenza",
    "Gas Sales Piacenza":                   "Piacenza",
    "Itas Trentino":                        "Trento",
    "Trentino Volley":                      "Trento",
    "Diatec Trentino":                      "Trento",
    "Allianz Milano":                       "Milano",
    "Powervolley Milano":                   "Milano",
    "Revivre Milano":                       "Milano",
    "Cucine Lube Civitanova":               "Lube",
    "Lube Civitanova":                      "Lube",
    "Rana Verona":                          "Verona",
    "Verona Volley":                        "Verona",
    "WithU Verona":                         "Verona",
    "NBV Verona":                           "Verona",
    "Calzedonia Verona":                    "Verona",
    "Kioene Padova":                        "Padova",
    "Sonepar Padova":                       "Padova",
    "Pallavolo Padova":                     "Padova",
    "Vero Volley Monza":                    "Monza",
    "Mint Vero Volley Monza":               "Monza",
    "Gi Group Monza":                       "Monza",
    "Valsa Group Modena":                   "Modena",
    "Leo Shoes Modena":                     "Modena",
    "Leo Shoes PerkinElmer Modena":         "Modena",
    "Azimut Modena":                        "Modena",
    "Azimut Leo Shoes Modena":              "Modena",
}

def normalize_club(name: str) -> str:
    s = str(name).strip()
    return CLUB_NAME_MAP.get(s, s)


# ═════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN DE DATOS
# ═════════════════════════════════════════════════════════════════

def load_sets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["equipo_local"]     = df["equipo_local"].apply(normalize_club)
    df["equipo_visitante"] = df["equipo_visitante"].apply(normalize_club)
    df = df[df["temporada"].isin(SEASONS)]
    df = df[df["equipo_local"].isin(TOP8) & df["equipo_visitante"].isin(TOP8)]
    return df


def build_match_table(df_sets: pd.DataFrame) -> pd.DataFrame:
    """Construye tabla de partidos con resultado global."""
    partidos = (
        df_sets.groupby("partido_id")
        .agg(
            local       = ("equipo_local",    "first"),
            visitante   = ("equipo_visitante","first"),
            temporada   = ("temporada",        "first"),
            jornada     = ("jornada",          "first"),
            sets_local  = ("ganador_set_local","sum"),
            total_sets  = ("set_num",          "count"),
        )
        .reset_index()
    )
    partidos["sets_visit"] = partidos["total_sets"] - partidos["sets_local"]
    partidos["gana_local"] = (partidos["sets_local"] > partidos["sets_visit"]).astype(int)

    def result_label(row):
        sl, sv = int(row["sets_local"]), int(row["sets_visit"])
        if sl > sv:
            return 0   # local gana (3-0, 3-1, 3-2)
        else:
            return 1   # visitante gana (0-3, 1-3, 2-3)

    partidos["result_idx"] = partidos.apply(result_label, axis=1)
    partidos["resultado"]  = partidos["result_idx"].map({0: "local", 1: "visitante"})
    # Ordenar cronológicamente dentro de cada temporada por jornada
    partidos["jornada_num"] = pd.to_numeric(
        partidos["jornada"].str.extract(r"(\d+)")[0], errors="coerce"
    ).fillna(0)
    partidos = partidos.sort_values(["temporada", "jornada_num"]).reset_index(drop=True)
    return partidos


def compute_team_history(partidos: pd.DataFrame, team: str, before_idx: int,
                          k: int = CONTEXT_LEN) -> np.ndarray | None:
    """
    Calcula la secuencia de los últimos k partidos del equipo
    anteriores al índice dado. Devuelve array (k, 5) o None si hay < 2 partidos.

    Features por partido:
        [0] set_win_rate      % sets ganados
        [1] pts_diff_avg      diferencia media de puntos por set
        [2] sets_largos_rate  % sets que fueron a prórroga
        [3] pts_avg_favor     puntos marcados por set
        [4] pts_avg_contra    puntos recibidos por set
    """
    hist = partidos[
        (partidos.index < before_idx) &
        ((partidos["local"] == team) | (partidos["visitante"] == team))
    ].tail(k)

    if len(hist) < 2:
        return None

    rows = []
    for _, h in hist.iterrows():
        is_local = h["local"] == team
        sl    = h["sets_local"] if is_local else h["sets_visit"]
        sv    = h["sets_visit"] if is_local else h["sets_local"]
        total = h["total_sets"]

        # Puntos: columnas añadidas en build_match_sequences via merge
        pf = float(h.get("pts_local_tot",  0) if is_local else h.get("pts_visit_tot", 0))
        pc = float(h.get("pts_visit_tot",  0) if is_local else h.get("pts_local_tot", 0))
        sl_rate    = float(h.get("sets_largos", 0)) / max(total, 1)
        pts_fav    = pf / max(total, 1)
        pts_con    = pc / max(total, 1)
        pts_diff   = (pf - pc) / max(total, 1)

        rows.append([
            sl / max(total, 1),   # set_win_rate
            pts_diff,             # pts_diff_avg
            sl_rate,              # sets_largos_rate
            pts_fav,              # pts_avg_favor
            pts_con,              # pts_avg_contra
        ])

    seq = np.array(rows, dtype=np.float32)

    # Pad/truncate a longitud k
    if len(seq) < k:
        pad = np.zeros((k - len(seq), 5), dtype=np.float32)
        seq = np.vstack([pad, seq])
    else:
        seq = seq[-k:]

    return seq


def build_match_sequences(partidos: pd.DataFrame,
                           df_sets: pd.DataFrame) -> list[dict]:
    """
    Para cada partido construye:
      - seq_local   (k, 5) — historial reciente del local
      - seq_visit   (k, 5) — historial reciente del visitante
      - result_idx  int    — índice del resultado (0-5)
      - gana_local  int    — 1/0
    """
    # Añadir pts_diff_avg y sets_largos a cada partido
    agg = df_sets.groupby("partido_id").agg(
        pts_local_tot  = ("puntos_local",      "sum"),
        pts_visit_tot  = ("puntos_visitante",   "sum"),
        sets_largos    = ("set_largo",          "sum"),
        total_sets     = ("set_num",            "count"),
    ).reset_index()
    partidos = partidos.merge(agg, on="partido_id", how="left",
                              suffixes=("", "_agg"))

    samples = []
    for idx, row in partidos.iterrows():
        local   = row["local"]
        visitante = row["visitante"]

        seq_l = compute_team_history(partidos, local,    idx)
        seq_v = compute_team_history(partidos, visitante, idx)
        if seq_l is None or seq_v is None:
            continue

        # Rellenar pts_diff y sets_largos en la última entrada (partido actual)
        # usando datos reales del partido — solo como señal de la secuencia histórica,
        # NO se usa información del partido a predecir
        samples.append({
            "seq_local":  seq_l,
            "seq_visit":  seq_v,
            "result_idx": int(row["result_idx"]) if not pd.isna(row["result_idx"]) else 0,
            "gana_local": int(row["gana_local"]),
        })

    return samples


# ═════════════════════════════════════════════════════════════════
# 2. FEATURES DE JUGADORES (Módulo 2)
# ═════════════════════════════════════════════════════════════════

def load_player_features() -> dict:
    """
    Carga los 5 archivos de jugadores y construye un dict:
      player_feats[(team, season)] = array (TOP_PLAYERS * 5,) = 25 features
    """
    frames = []
    for key, path in PLAYER_FILES.items():
        if not os.path.exists(path):
            print(f"  ⚠️  No encontrado: {path}")
            continue
        col = PLAYER_STAT_COLS[key]
        df  = pd.read_excel(path)
        df["team_norm"] = df["Team"].apply(normalize_club)
        df["season"]    = df["Season"].astype(str)
        tmp = df[["Player","team_norm","season","Played Sets", col]].copy()
        tmp.columns = ["player","team","season","sets_played","stat_val"]
        tmp["stat"]   = key
        tmp["sets_played"] = pd.to_numeric(tmp["sets_played"], errors="coerce").fillna(1)
        tmp["stat_val"]    = pd.to_numeric(tmp["stat_val"],    errors="coerce").fillna(0)
        frames.append(tmp)

    if not frames:
        return {}

    all_stats = pd.concat(frames, ignore_index=True)
    all_stats  = all_stats[all_stats["team"].isin(TOP8)]

    player_feats = {}
    stat_keys    = list(PLAYER_STAT_COLS.keys())

    for (team, season), grp in all_stats.groupby(["team","season"]):
        # Media ponderada por sets jugados para cada (jugador, stat)
        rows = []
        for (player, stat), g in grp.groupby(["player","stat"]):
            w = g["sets_played"].values
            v = g["stat_val"].values
            rows.append({"player": player, "stat": stat,
                         "val": np.average(v, weights=w) if w.sum() > 0 else 0.0})

        if not rows:
            continue

        pivot = pd.DataFrame(rows).pivot(
            index="player", columns="stat", values="val"
        ).fillna(0).reindex(columns=stat_keys, fill_value=0)

        # Score global para ranking
        pivot_norm = pivot.copy()
        for col in pivot_norm.columns:
            mn, mx = pivot_norm[col].min(), pivot_norm[col].max()
            pivot_norm[col] = (pivot_norm[col] - mn) / (mx - mn) if mx > mn else 0.0
        pivot["score"] = pivot_norm.mean(axis=1)

        top = pivot.sort_values("score", ascending=False).head(TOP_PLAYERS)
        top = top.drop(columns="score")

        # Pad si hay menos de TOP_PLAYERS jugadores
        while len(top) < TOP_PLAYERS:
            top = pd.concat([top, pd.DataFrame(
                [[0.0] * len(stat_keys)], columns=stat_keys
            )], ignore_index=True)

        player_feats[(team, season)] = top[stat_keys].values.flatten().astype(np.float32)

    return player_feats


def build_set_samples(df_sets: pd.DataFrame,
                       player_feats: dict) -> list[dict]:
    """
    Para cada set construye:
      - feats_local  (25,) — stats de los 5 jugadores del local esa temporada
      - feats_visit  (25,) — stats de los 5 jugadores del visitante
      - gana_local   int
    """
    # Pre-calcular marcador parcial antes de cada set agrupando por partido
    df_sets = df_sets.sort_values(["partido_id","set_num"]).copy()
    df_sets["sl_antes"] = df_sets.groupby("partido_id")["ganador_set_local"].cumsum() - df_sets["ganador_set_local"]
    df_sets["sv_antes"] = (df_sets["set_num"] - 1) - df_sets["sl_antes"]

    samples = []
    for _, row in df_sets.iterrows():
        local    = row["equipo_local"]
        visitante = row["equipo_visitante"]
        season   = row["temporada"]

        fl = player_feats.get((local,    season))
        fv = player_feats.get((visitante, season))
        if fl is None or fv is None:
            continue

        sl_antes = float(row["sl_antes"])
        sv_antes = float(row["sv_antes"])
        set_num  = float(row["set_num"])
        # 4 features de contexto: set_num/5, sets_local/3, sets_visit/3, diff_parcial/2
        ctx = np.array([
            set_num / 5.0,
            sl_antes / 3.0,
            sv_antes / 3.0,
            (sl_antes - sv_antes) / 2.0,
        ], dtype=np.float32)

        samples.append({
            "feats_local":  fl,
            "feats_visit":  fv,
            "ctx":          ctx,
            "gana_local":   int(row["ganador_set_local"]),
        })

    return samples


# ═════════════════════════════════════════════════════════════════
# 3. DATASETS PYTORCH
# ═════════════════════════════════════════════════════════════════

class MatchDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["seq_local"],  dtype=torch.float32),
            torch.tensor(s["seq_visit"],  dtype=torch.float32),
            torch.tensor(s["result_idx"], dtype=torch.long),
        )


class SetDataset(Dataset):
    def __init__(self, samples: list[dict], scaler: StandardScaler = None,
                 fit_scaler: bool = False):
        X = np.stack([
            np.concatenate([s["feats_local"], s["feats_visit"],
                            s.get("ctx", np.array([0.2, 0., 0., 0.]))])
            for s in samples
        ])
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = scaler.transform(X) if scaler else X
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(
            [s["gana_local"] for s in samples], dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═════════════════════════════════════════════════════════════════
# 4. ARQUITECTURAS TRANSFORMER
# ═════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal estándar."""
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TeamEncoder(nn.Module):
    """
    Codifica la secuencia de partidos de un equipo en un embedding fijo.
    Arquitectura: proyección lineal → PE → Transformer Encoder → mean pooling
    """
    def __init__(self, input_dim: int = 5, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(input_dim, d_model)
        self.pe      = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.proj(x)          # (batch, seq_len, d_model)
        x = self.pe(x)
        x = self.encoder(x)       # (batch, seq_len, d_model)
        return x.mean(dim=1)      # (batch, d_model) — mean pooling


class MatchTransformer(nn.Module):
    """
    Módulo 1: predice el resultado del partido a partir de las secuencias
    de historial reciente de ambos equipos.

    Salida: logits sobre 6 clases (3-0, 3-1, 3-2, 0-3, 1-3, 2-3)
    """
    def __init__(self, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1,
                 num_classes: int = 6):
        super().__init__()
        self.encoder_local = TeamEncoder(5, d_model, nhead, num_layers, dropout)
        self.encoder_visit = TeamEncoder(5, d_model, nhead, num_layers, dropout)

        # Cross-attention: local atiende al visitante
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, seq_local: torch.Tensor,
                seq_visit: torch.Tensor) -> torch.Tensor:
        emb_l = self.encoder_local(seq_local)   # (batch, d_model)
        emb_v = self.encoder_visit(seq_visit)   # (batch, d_model)

        # Cross-attention: local como query, visitante como key/value
        emb_l_3d  = emb_l.unsqueeze(1)          # (batch, 1, d_model)
        emb_v_3d  = emb_v.unsqueeze(1)
        attn_out, _ = self.cross_attn(emb_l_3d, emb_v_3d, emb_v_3d)
        emb_l = self.norm(emb_l + attn_out.squeeze(1))

        combined = torch.cat([emb_l, emb_v], dim=-1)  # (batch, d_model*2)
        return self.head(combined)


class SetTransformer(nn.Module):
    """
    Módulo 2: predice el ganador de un set a partir de las stats
    de los 5 jugadores clave de cada equipo (25 features × 2 = 50).

    El transformer trata las 5 estadísticas de cada jugador como una
    secuencia de tokens (10 tokens en total: 5 local + 5 visitante).
    """
    def __init__(self, n_players: int = 5, n_stats: int = 5,
                 d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_players = n_players
        self.n_stats   = n_stats

        # Proyectar cada jugador (n_stats features) a d_model
        self.proj = nn.Linear(n_stats, d_model)
        self.pe   = PositionalEncoding(d_model, max_len=n_players * 2, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # +4 features de contexto: set_num, sl_antes, sv_antes, diff_parcial
        self.head = nn.Sequential(
            nn.Linear(d_model * n_players * 2 + 4, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 54) — 25 local + 25 visitante + 4 contexto
        batch   = x.size(0)
        ctx     = x[:, -4:]                           # (batch, 4) — set_num, sl, sv, diff
        players = x[:, :-4]                           # (batch, 50)
        players = players.view(batch, self.n_players * 2, self.n_stats)
        players = self.proj(players)
        players = self.pe(players)
        players = self.encoder(players)
        flat    = players.reshape(batch, -1)           # (batch, 10*d_model)
        combined = torch.cat([flat, ctx], dim=-1)      # (batch, 10*d_model + 4)
        return self.head(combined)


# ═════════════════════════════════════════════════════════════════
# 5. ENTRENAMIENTO
# ═════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        optimizer.zero_grad()
        if len(batch) == 3:          # MatchDataset
            sl, sv, y = [b.to(device) for b in batch]
            logits = model(sl, sv)
        else:                        # SetDataset
            x, y = [b.to(device) for b in batch]
            logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                sl, sv, y = [b.to(device) for b in batch]
                logits = model(sl, sv)
            else:
                x, y = [b.to(device) for b in batch]
                logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            total      += y.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, epochs: int,
                lr: float, device, name: str, class_weights=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()
    best_val_acc, best_state = 0.0, None

    print(f"\n  Entrenando {name} ({epochs} épocas)...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Época {epoch:>3}  |  "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.3f}  |  "
                  f"val acc {va_acc:.3f}  {'← mejor' if va_acc == best_val_acc else ''}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Mejor val acc: {best_val_acc:.3f}")
    return model


# ═════════════════════════════════════════════════════════════════
# 6. CONTENEDOR UNIFICADO
# ═════════════════════════════════════════════════════════════════

class VBTransformerPredictor:
    """
    Contenedor que agrupa MatchTransformer + SetTransformer
    y guarda todo lo necesario para predecir sin reentrenar.
    """
    def __init__(self):
        self.match_model   = None
        self.set_model     = None
        self.set_scaler    = None
        self.player_feats  = {}   # {(team, season): array(25,)}
        self.match_history = None  # DataFrame de partidos para contexto

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Guardar pesos de los modelos separados (compatibilidad joblib + torch)
        state = {
            "match_state":    self.match_model.state_dict() if self.match_model else None,
            "set_state":      self.set_model.state_dict()   if self.set_model   else None,
            "set_scaler":     self.set_scaler,
            "player_feats":   self.player_feats,
            "match_history":  self.match_history,
            "match_config":   self._match_cfg,
            "set_config":     self._set_cfg,
        }
        joblib.dump(state, path)
        print(f"  ✅ Guardado: '{path}'  ({os.path.getsize(path)//1024} KB)")

    @classmethod
    def load(cls, path: str) -> "VBTransformerPredictor":
        state = joblib.load(path)
        pred  = cls()
        pred.player_feats  = state["player_feats"]
        pred.match_history = state["match_history"]
        pred.set_scaler    = state["set_scaler"]

        if state["match_state"] and state["match_config"]:
            cfg = state["match_config"]
            pred.match_model = MatchTransformer(**cfg)
            pred.match_model.load_state_dict(state["match_state"])
            pred.match_model.eval()

        if state["set_state"] and state["set_config"]:
            cfg = state["set_config"]
            pred.set_model = SetTransformer(**cfg)
            pred.set_model.load_state_dict(state["set_state"])
            pred.set_model.eval()

        return pred


# ═════════════════════════════════════════════════════════════════
# 7. PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PREDICTOR VB — Entrenamiento Transformer")
    print("=" * 60)
    print(f"  Equipos  : {TOP8}")
    print(f"  Temporadas: {SEASONS}")
    print(f"  Dispositivo: {DEVICE}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Cargar datos ───────────────────────────────────────────────
    print("\n  Cargando datos...")
    df_sets  = load_sets(SETS_CSV)
    partidos = build_match_table(df_sets)
    print(f"  Partidos: {len(partidos)}  |  Sets: {len(df_sets)}")

    # ── Módulo 1: Match Transformer ────────────────────────────────
    print("\n" + "─" * 60)
    print("  MÓDULO 1 — Match Transformer")
    print("─" * 60)

    match_samples = build_match_sequences(partidos, df_sets)
    print(f"  Muestras de entrenamiento: {len(match_samples)}")

    if len(match_samples) < 20:
        print("  ⚠️  Muy pocas muestras para entrenar el Match Transformer.")
    else:
        tr_s, va_s = train_test_split(match_samples, test_size=0.2, random_state=42)
        tr_loader  = DataLoader(MatchDataset(tr_s), batch_size=16, shuffle=True)
        va_loader  = DataLoader(MatchDataset(va_s), batch_size=16)

        match_cfg = dict(d_model=64, nhead=4, num_layers=2, dropout=0.1, num_classes=3)
        match_model = MatchTransformer(**match_cfg).to(DEVICE)

        # Pesos de clase inversos a la frecuencia: local_win=86%, visit_win=86% → peso bajo
        # El modelo debe aprender a distinguir, no solo predecir la clase mayoritaria.
        # Calculamos pesos a partir de los datos reales.
        labels = [s["result_idx"] for s in tr_s]
        counts = np.bincount(labels, minlength=3).astype(float)
        counts = np.where(counts == 0, 1, counts)
        weights = (len(labels) / (len(counts) * counts)).tolist()
        print(f"  Pesos de clase: {[round(w,2) for w in weights]}")

        match_model = train_model(
            match_model, tr_loader, va_loader,
            epochs=100, lr=5e-4, device=DEVICE, name="MatchTransformer",
            class_weights=weights,
        )

    # ── Módulo 2: Set Transformer ──────────────────────────────────
    print("\n" + "─" * 60)
    print("  MÓDULO 2 — Set Transformer")
    print("─" * 60)

    player_feats = load_player_features()
    print(f"  Combinaciones (equipo, temporada) con datos: {len(player_feats)}")

    set_samples = build_set_samples(df_sets, player_feats)
    print(f"  Muestras de sets: {len(set_samples)}")

    set_model, set_scaler = None, None

    if len(set_samples) < 50:
        print("  ⚠️  Muy pocas muestras para entrenar el Set Transformer.")
    else:
        tr_s, va_s = train_test_split(set_samples, test_size=0.2, random_state=42)
        tr_ds = SetDataset(tr_s, fit_scaler=True)
        va_ds = SetDataset(va_s, scaler=tr_ds.scaler)
        set_scaler = tr_ds.scaler

        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=32)

        set_cfg = dict(n_players=TOP_PLAYERS, n_stats=5,
                       d_model=16, nhead=2, num_layers=1, dropout=0.1)
        set_model = SetTransformer(**set_cfg).to(DEVICE)

        labels = [s["gana_local"] for s in tr_s]
        counts = np.bincount(labels, minlength=2).astype(float)
        counts = np.where(counts == 0, 1, counts)

        weights = (len(labels) / (len(counts) * counts)).tolist()

        print(f"  Pesos de clase sets: {[round(w,2) for w in weights]}")

        set_model = train_model(
            set_model,
            tr_loader,
            va_loader,
            epochs=80,
            lr=1e-3,
            device=DEVICE,
            name="SetTransformer",
            class_weights=weights,
)

    # ── Guardar ────────────────────────────────────────────────────
    vb = VBTransformerPredictor()
    vb.match_model   = match_model if len(match_samples) >= 20 else None
    vb.set_model     = set_model
    vb.set_scaler    = set_scaler
    vb.player_feats  = player_feats
    vb.match_history = partidos
    vb._match_cfg    = match_cfg if len(match_samples) >= 20 else None
    vb._set_cfg      = set_cfg   if set_model else None

    vb.save(os.path.join(MODEL_DIR, "vb_transformer.pkl"))

    print("\n" + "=" * 60)
    print("  Entrenamiento completado.")
    print(f"  Modelo guardado en: {MODEL_DIR}/vb_transformer.pkl")
    print("  Ejecuta predictor.py para hacer predicciones.")
    print("=" * 60)