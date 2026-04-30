"""
predecir.py
───────────
Carga un modelo entrenado hasta una temporada concreta y simula el partido
set a set, mostrando para cada set:
  - Probabilidad de victoria de cada equipo
  - Marcador estimado con rango (±4 puntos)
  - Cómo evoluciona la probabilidad de ganar el partido

Los modelos se generan con entrenar.py.

¿Qué modelo usar?
    → Para un partido de 2024/25 usa modelo_hasta_2024_2025.pkl
      (el modelo solo conoce información de temporadas anteriores)
    → Para la mejor predicción posible usa modelo_hasta_2025_2026.pkl

Edita la sección CONFIGURACIÓN y ejecuta:
    python predecir.py
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — Edita aquí tu predicción
# ═════════════════════════════════════════════════════════════════

# Opciones: 2021_2022 | 2022_2023 | 2023_2024 | 2024_2025 | 2025_2026
MODELO_HASTA = "2024_2025"

HOME_CLUB = "Perugia"
AWAY_CLUB = "Trento"
SEASON    = "2024/2025"


# ═════════════════════════════════════════════════════════════════
# NORMALIZACIÓN (debe coincidir con entrenar.py)
# ═════════════════════════════════════════════════════════════════

CLUB_NAME_MAP = {
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

CLUB_SEASON_MAP = {
    ("Cisterna", "2020/2021"): "Cisterna Top Volley",
    ("Cisterna", "2021/2022"): "Cisterna Top Volley",
    ("Cisterna", "2022/2023"): "Cisterna Top Volley",
}

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


# ═════════════════════════════════════════════════════════════════
# CLASES STUB — necesarias para que joblib deserialice el .pkl
# ═════════════════════════════════════════════════════════════════
# No contienen lógica propia: toda la predicción está en los objetos
# cargados desde el .pkl (entrenados por entrenar.py).

class MatchPredictor:
    pass

class SetBySetPredictor:
    pass

class VBPredictor:
    @classmethod
    def load(cls, path: str) -> "VBPredictor":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Modelo no encontrado: '{path}'\n"
                f"Ejecuta entrenar.py para generarlo."
            )
        return joblib.load(path)


# ═════════════════════════════════════════════════════════════════
# SIMULACIÓN DE PARTIDO
# ═════════════════════════════════════════════════════════════════

def prob_partido(prob_set_local: float, sets_local: int, sets_visitante: int) -> float:
    """
    Calcula la probabilidad de que el equipo local gane el partido
    dado el marcador actual y la probabilidad de ganar el siguiente set.

    Usa recursión sobre los posibles estados restantes del partido.
    El primero en llegar a 3 sets gana (formato al mejor de 5).
    """
    # Casos base
    if sets_local == 3:
        return 1.0
    if sets_visitante == 3:
        return 0.0
    # Sets que le faltan a cada equipo para ganar
    faltan_local     = 3 - sets_local
    faltan_visitante = 3 - sets_visitante
    total_restantes  = faltan_local + faltan_visitante - 1  # sets que se jugarán como mínimo

    # Suma ponderada sobre todos los posibles caminos restantes
    p = prob_set_local
    q = 1 - p
    prob = 0.0
    # El local gana si consigue `faltan_local` sets en los próximos
    # `faltan_local + faltan_visitante - 1` sets
    from math import comb
    for wins in range(faltan_local, faltan_local + faltan_visitante):
        # El local gana exactamente `wins` sets en `wins + faltan_visitante - 1` intentos
        # y el último es victoria del local
        n = wins + faltan_visitante - 1
        k = wins - 1
        if k < 0 or k > n:
            continue
        prob += comb(n, k) * (p ** (k)) * (q ** (n - k)) * p
    return min(max(prob, 0.0), 1.0)


def barra_prob(prob: float, ancho: int = 24) -> str:
    """Genera una barra visual de probabilidad para la consola."""
    llenos = round(prob * ancho)
    return "█" * llenos + "░" * (ancho - llenos)


def simular_partido(vb: VBPredictor, home: str, away: str, season: str) -> None:
    """
    Simula un partido set a set de forma automática, mostrando
    para cada set la predicción y cómo evoluciona la prob de victoria.
    """
    home = normalize_club(home, season)
    away = normalize_club(away, season)

    W = 60  # ancho de la salida

    print(f"\n{'═' * W}")
    print(f"  {home}  vs  {away}".center(W))
    print(f"  Temporada: {season}  |  Modelo: hasta {MODELO_HASTA.replace('_','/')}".center(W))
    print(f"{'═' * W}")

    # ── Predicción global del partido ─────────────────────────────
    match_res = vb.match_model.predict(home, away, season)
    hs = match_res["home_sets_rounded"]
    vs = match_res["away_sets_rounded"]
    hc = match_res["home_sets"]
    vc = match_res["away_sets"]

    ganador_global = home if hs > vs else (away if vs > hs else "Empate técnico")
    print(f"\n  RESULTADO GLOBAL PREDICHO")
    print(f"  {'─' * (W-2)}")
    print(f"  {home:<28} {hs} sets  (valor continuo: {hc})")
    print(f"  {away:<28} {vs} sets  (valor continuo: {vc})")
    print(f"  → {ganador_global}")

    # ── Simulación set a set ──────────────────────────────────────
    # Determinar cuántos sets se jugarán según la predicción global
    total_sets = hs + vs

    # Si el resultado global no es válido (ej: 2-2), forzamos un resultado coherente
    if hs == vs or (hs < 3 and vs < 3):
        # Usar el valor continuo para desempatar
        if hc >= vc:
            hs, vs = 3, max(0, min(2, round(vc)))
        else:
            vs, hs = 3, max(0, min(2, round(hc)))
        total_sets = hs + vs

    sets_local     = 0
    sets_visitante = 0

    print(f"\n  SIMULACIÓN SET A SET")
    print(f"  {'─' * (W-2)}")

    for set_num in range(1, total_sets + 1):

        # Predecir este set
        pred = vb.set_model.predict_set(
            local=home,
            visitante=away,
            season=season,
            set_num=set_num,
            sets_local_antes=sets_local,
            sets_visitante_antes=sets_visitante,
        )

        pl  = pred["prob_local"]
        pv  = pred["prob_visitante"]
        gl  = pred["gana_local"]
        ptl = pred["pts_local_est"]
        ptv = pred["pts_visit_est"]
        lmin, lmax = pred["pts_local_min"], pred["pts_local_max"]
        vmin, vmax = pred["pts_visit_min"], pred["pts_visit_max"]

        # Probabilidad de ganar el partido ANTES de este set
        prob_antes = prob_partido(pl, sets_local, sets_visitante)

        # Actualizar marcador según predicción
        if gl:
            sets_local += 1
        else:
            sets_visitante += 1

        # Probabilidad de ganar el partido DESPUÉS de este set
        prob_despues = prob_partido(pl, sets_local, sets_visitante)

        # Flecha de tendencia
        delta = prob_despues - prob_antes
        if   delta >  0.05: tendencia = "↑"
        elif delta < -0.05: tendencia = "↓"
        else:                tendencia = "→"

        ganador_set = home if gl else away

        print(f"\n  SET {set_num}  [{sets_local - (1 if gl else 0)}"
              f"-{sets_visitante - (0 if gl else 1)} antes]")
        print(f"  {'─' * (W-2)}")
        print(f"  Ganador predicho : {ganador_set}")
        print(f"  Marcador est.    : {ptl}-{ptv}  "
              f"(rango: {max(0,lmin)}-{max(0,vmin)}  a  {lmax}-{vmax})")
        print()
        print(f"  {home[:22]:<22} {pl*100:>5.1f}%  {barra_prob(pl)}")
        print(f"  {away[:22]:<22} {pv*100:>5.1f}%  {barra_prob(pv)}")
        print()
        print(f"  Prob. ganar partido → {home}: {prob_despues*100:.1f}%  "
              f"{tendencia}  (antes: {prob_antes*100:.1f}%)")

        # Marcador acumulado
        print(f"  Marcador parcial : {home} {sets_local} - {sets_visitante} {away}")

    # ── Resultado final ────────────────────────────────────────────
    ganador_final = home if sets_local > sets_visitante else away
    print(f"\n{'═' * W}")
    print(f"  RESULTADO FINAL PREDICHO".center(W))
    print(f"  {home} {sets_local}  —  {sets_visitante} {away}".center(W))
    print(f"  Ganador: {ganador_final}".center(W))
    print(f"{'═' * W}\n")


# ═════════════════════════════════════════════════════════════════
# UTILIDADES
# ═════════════════════════════════════════════════════════════════

def listar_modelos(model_dir: str) -> list:
    modelos = sorted(glob.glob(os.path.join(model_dir, "modelo_hasta_*.pkl")))
    if not modelos:
        print("⚠️  No hay modelos en model/. Ejecuta entrenar.py primero.")
    else:
        print("Modelos disponibles:")
        for m in modelos:
            print(f"  {os.path.basename(m)}  ({os.path.getsize(m)//1024} KB)")
    return modelos


# ═════════════════════════════════════════════════════════════════
# ENTRADA PRINCIPAL
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PREDICTOR VB — Simulación set a set")
    print("=" * 60)

    listar_modelos("model")

    model_path = os.path.join("model", f"modelo_hasta_{MODELO_HASTA}.pkl")
    vb = VBPredictor.load(model_path)

    simular_partido(vb, HOME_CLUB, AWAY_CLUB, SEASON)