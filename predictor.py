"""
predecir.py
───────────
Carga un modelo entrenado hasta una temporada concreta y predice el resultado
de un partido. Los modelos se generan con entrenar.py.

¿Qué modelo usar?
    → Para un partido de 2024/25 usa modelo_hasta_2024_2025.pkl
      (el modelo solo conoce información de temporadas anteriores, sin data leakage)
    → Para la mejor predicción posible con todos los datos, usa
      modelo_hasta_2025_2026.pkl

Edita la sección CONFIGURACIÓN y ejecuta:
    python predecir.py
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")



# CONFIGURACIÓN

# Opciones: 2021_2022 | 2022_2023 | 2023_2024 | 2024_2025 | 2025_2026
MODELO_HASTA = "2024_2025"

HOME_CLUB = "Verona"
AWAY_CLUB = "Modena"
SEASON    = "2024/2025"   # Temporada de las stats a usar


# ═════════════════════════════════════════════
# NORMALIZACIÓN DE NOMBRES (igual que entrenar.py)
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
# FEATURE COLS (debe coincidir con entrenar.py)
# ═════════════════════════════════════════════

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


# ═════════════════════════════════════════════
# CLASE PREDICTOR (necesaria para cargar el .pkl)
# ═════════════════════════════════════════════

class SetPredictor:
    """Replica mínima para poder cargar el pkl y llamar a predict()."""

    def predict(self, home_club: str, away_club: str, season: str) -> dict:
        if self._team_feat is None:
            raise RuntimeError("El modelo no tiene features de equipo almacenadas.")

        home_club = normalize_club(home_club, season)
        away_club = normalize_club(away_club, season)

        def get_vector(club):
            row = self._team_feat[
                (self._team_feat["club"] == club) &
                (self._team_feat["season"] == season)
            ]
            if row.empty:
                return pd.Series({c: np.nan for c in FEATURE_COLS})
            available = [c for c in FEATURE_COLS if c in row.columns]
            result = pd.Series({c: np.nan for c in FEATURE_COLS})
            result.update(row[available].iloc[0])
            return result

        diff = (get_vector(home_club) - get_vector(away_club)).add_prefix("diff_")
        X = pd.DataFrame([diff])[self.diff_cols].fillna(0)
        X_scaled = self.scaler.transform(X)

        home_pred = float(self.model_home.predict(X_scaled)[0])
        away_pred = float(self.model_away.predict(X_scaled)[0])

        return {
            "home_sets":         round(home_pred, 2),
            "away_sets":         round(away_pred, 2),
            "home_sets_rounded": int(np.clip(round(home_pred), 0, 3)),
            "away_sets_rounded": int(np.clip(round(away_pred), 0, 3)),
        }

    @classmethod
    def load(cls, path: str) -> "SetPredictor":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Modelo no encontrado: '{path}'\n"
                f"Ejecuta entrenar.py para generarlo."
            )
        return joblib.load(path)


# ═════════════════════════════════════════════
# PREDICCIÓN
# ═════════════════════════════════════════════

def listar_modelos(model_dir: str) -> list:
    modelos = sorted(glob.glob(os.path.join(model_dir, "modelo_hasta_*.pkl")))
    if not modelos:
        print("⚠️  No hay modelos en model/. Ejecuta entrenar.py primero.")
    else:
        print("Modelos disponibles:")
        for m in modelos:
            print(f"  {os.path.basename(m)}  ({os.path.getsize(m)//1024} KB)")
    return modelos


def predecir(model_path: str, home: str, away: str, season: str) -> None:
    print(f"\n{'─' * 52}")
    print(f"  Modelo  : {os.path.basename(model_path)}")
    print(f"  Partido : {home} (local) vs {away} (visitante)")
    print(f"  Stats   : temporada {season}")
    print(f"{'─' * 52}")

    model  = SetPredictor.load(model_path)
    result = model.predict(home_club=home, away_club=away, season=season)

    hs  = result["home_sets_rounded"]
    as_ = result["away_sets_rounded"]
    hc  = result["home_sets"]
    ac  = result["away_sets"]

    ganador = f"Gana {home}" if hs > as_ else (
              f"Gana {away}" if as_ > hs else
              "Empate técnico (revisar — no válido en voleibol)")

    print(f"\n  {home:<28} {hs} sets  (continuo: {hc})")
    print(f"  {away:<28} {as_} sets  (continuo: {ac})")
    print(f"\n  >>> {ganador}")
    print(f"{'─' * 52}\n")


if __name__ == "__main__":
    print("=" * 52)
    print("  PREDICTOR VB — Predicción de partido")
    print("=" * 52)

    listar_modelos("model")
    predecir(
        model_path=os.path.join("model", f"modelo_hasta_{MODELO_HASTA}.pkl"),
        home=HOME_CLUB,
        away=AWAY_CLUB,
        season=SEASON,
    )