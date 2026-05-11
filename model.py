"""
model.py
--------
TabTransformer-style model for volleyball match prediction.

Architecture:
  - Categorical embeddings for team identities
  - Numeric feature projection to d_model dimension
  - N Transformer encoder layers over the feature tokens
  - Two prediction heads:
      * match_result: 6-class (3-0, 3-1, 3-2, 0-3, 1-3, 2-3)
      * local_wins:   binary (home team wins?)

The model treats each feature group (form, h2h, team_stats, player_stats,
context) as a "token" in the sequence, allowing attention to capture
cross-feature interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """
    Projects each numeric feature to d_model via a shared linear layer,
    then stacks them as a sequence of tokens.
    
    Shape in:  (B, n_features)
    Shape out: (B, n_features, d_model)
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)  →  (B, F, 1)  →  (B, F, d_model)
        return self.proj(x.unsqueeze(-1))


class TeamEmbedding(nn.Module):
    """Learnable embedding for team identity."""

    def __init__(self, n_teams: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(n_teams + 1, d_model, padding_idx=0)

    def forward(self, team_idx: torch.Tensor) -> torch.Tensor:
        return self.emb(team_idx)  # (B, d_model)


class VolleyballTransformer(nn.Module):
    """
    Transformer encoder for volleyball match outcome prediction.

    Parameters
    ----------
    n_numeric_features : int
        Number of numeric input features.
    n_teams : int
        Number of unique teams in the dataset.
    d_model : int
        Transformer embedding dimension.
    n_heads : int
        Number of attention heads (must divide d_model).
    n_layers : int
        Number of Transformer encoder layers.
    d_ff : int
        Feed-forward hidden dimension inside Transformer.
    dropout : float
        Dropout probability.
    n_result_classes : int
        Number of output classes for match result (default 6).
    """

    def __init__(
        self,
        n_numeric_features: int,
        n_teams: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_result_classes: int = 6,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Input layers ---
        self.feature_tokenizer = FeatureTokenizer(n_numeric_features, d_model)
        self.local_team_emb = TeamEmbedding(n_teams, d_model)
        self.visit_team_emb = TeamEmbedding(n_teams, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding (learned, not sinusoidal – simpler for tabular)
        # Sequence length = 1 (CLS) + n_numeric_features + 2 (teams)
        seq_len = 1 + n_numeric_features + 2
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Output heads ---
        self.dropout = nn.Dropout(dropout)

        # Match result head (6-class)
        self.result_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_result_classes),
        )

        # Local wins head (binary – uses sigmoid externally via BCEWithLogitsLoss)
        self.wins_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        numeric: torch.Tensor,       # (B, n_numeric_features)
        local_idx: torch.Tensor,     # (B,)
        visit_idx: torch.Tensor,     # (B,)
    ) -> dict:
        B = numeric.size(0)

        # Feature tokens: (B, n_features, d_model)
        feat_tokens = self.feature_tokenizer(numeric)

        # Team tokens: (B, 1, d_model)
        local_tok = self.local_team_emb(local_idx).unsqueeze(1)
        visit_tok = self.visit_team_emb(visit_idx).unsqueeze(1)

        # CLS token: (B, 1, d_model)
        cls = self.cls_token.expand(B, -1, -1)

        # Concatenate: [CLS, local_team, visit_team, feat_1, ..., feat_F]
        x = torch.cat([cls, local_tok, visit_tok, feat_tokens], dim=1)

        # Add positional embeddings
        x = x + self.pos_emb

        # Transformer encoder
        x = self.transformer(x)

        # Use CLS token representation for prediction
        cls_out = self.dropout(x[:, 0, :])  # (B, d_model)

        result_logits = self.result_head(cls_out)     # (B, 6)
        wins_logit = self.wins_head(cls_out).squeeze(-1)  # (B,)

        return {
            "result_logits": result_logits,
            "wins_logit": wins_logit,
        }


class VolleyballLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    
    L = alpha * CE(result) + (1-alpha) * BCE(wins)
    
    The result loss is weighted to penalise direction errors
    (predicting local win when away wins) more than magnitude errors.
    """

    def __init__(self, alpha: float = 0.7, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.result_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.wins_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict, targets: dict) -> dict:
        result_l = self.result_loss(outputs["result_logits"], targets["match_result"])
        wins_l = self.wins_loss(outputs["wins_logit"], targets["local_wins"].float())

        total = self.alpha * result_l + (1 - self.alpha) * wins_l

        return {
            "total": total,
            "result": result_l,
            "wins": wins_l,
        }
