"""
predictor.py
────────────
Carga un modelo entrenado con trainer.py y simula un partido completo set a set.

Edita la sección CONFIGURACIÓN y ejecuta:
    python predictor.py
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Importar las clases desde trainer.py (en la misma carpeta) ──
# IMPORTANTE: esto es lo que permite que joblib deserialice el .pkl correctamente.
# Las clases deben ser las mismas que se usaron al entrenar.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainer import (
    VBPredictor, MatchPredictor, SetBySetPredictor,
    normalize_club, FEATURE_COLS,
)


# ═════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — Edita aquí
# ═════════════════════════════════════════════════════════════════

MODELO_HASTA = "2024_2025"   # 2021_2022 | 2022_2023 | 2023_2024 | 2024_2025 | 2025_2026

HOME_CLUB = "Verona"
AWAY_CLUB = "Milano"
SEASON    = "2024/2025"


# ═════════════════════════════════════════════════════════════════
# JUGADORES DESTACADOS
# ═════════════════════════════════════════════════════════════════

def mostrar_jugadores_destacados(
    model: VBPredictor,
    home: str,
    away: str,
    season: str,
    top_n: int = 3,
):
    paths = glob.glob("DB/stats_por_equipo_completo/*_historial_10_años.csv")
    if not paths:
        print("  (archivos de jugadores no encontrados)")
        return

    frames = []
    for p in paths:
        try:
            df    = pd.read_csv(p)
            fname = os.path.basename(p)
            team  = fname.replace("_historial_10_años.csv", "").replace("_", " ")
            df["equipo"] = team
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return

    df_all = pd.concat(frames, ignore_index=True)
    df_all.columns = df_all.columns.str.strip()
    df_all = df_all.rename(columns={
        "Player_Player":        "jugador",
        "Temporada":            "temporada",
        "ATTACK_Effic.":        "att_effic",
        "RECEPTION_Effic.":     "rec_effic",
        "BLOCK_Points per Set": "block_per_set",
        "SERVE_Ace per Set":    "serve_ace_per_set",
        "Played Set_Played Set":"sets_jugados",
    })

    def fix_s(x):
        x = str(x)
        return x if "/" in x else f"{x}/{int(x)+1}"
    if "temporada" in df_all.columns:
        df_all["temporada"] = df_all["temporada"].apply(fix_s)
    df_all["equipo"] = df_all["equipo"].apply(normalize_club)

    categorias = {
        "Ataque":    ("att_effic",         "Efic. ataque"),
        "Recepción": ("rec_effic",         "Efic. recepción"),
        "Bloqueo":   ("block_per_set",     "Bloqueos/set"),
        "Saque":     ("serve_ace_per_set", "Aces/set"),
    }

    for equipo_nombre in [home, away]:
        equipo_norm = normalize_club(equipo_nombre, season)
        print(f"\n  ── {equipo_nombre} ──")
        df_eq = df_all[
            (df_all["equipo"] == equipo_norm) &
            (df_all.get("temporada", pd.Series(dtype=str)) == season)
        ] if "temporada" in df_all.columns else df_all[df_all["equipo"] == equipo_norm]

        if df_eq.empty:
            print(f"    (sin datos para {season})")
            continue

        if "sets_jugados" in df_eq.columns:
            df_eq = df_eq[
                pd.to_numeric(df_eq["sets_jugados"], errors="coerce").fillna(0) >= 5
            ]

        for cat, (col, desc) in categorias.items():
            if col not in df_eq.columns:
                continue
            tmp = df_eq[["jugador", col]].copy() if "jugador" in df_eq.columns else df_eq[[col]].copy()
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            tmp = tmp.dropna(subset=[col]).sort_values(col, ascending=False).head(top_n)
            if tmp.empty:
                continue
            print(f"    {cat} ({desc}):")
            for _, r in tmp.iterrows():
                nombre = r.get("jugador", "—") if "jugador" in r.index else "—"
                print(f"      {nombre:<30} {round(r[col], 3):>8}")


# ═════════════════════════════════════════════════════════════════
# PROBABILIDAD DE PARTIDO
# ═════════════════════════════════════════════════════════════════

def prob_partido(prob_set_local: float, sets_local: int, sets_visitante: int) -> float:
    """Calcula P(local gana partido) dado marcador parcial y prob de ganar cada set."""
    from math import comb
    if sets_local == 3:   return 1.0
    if sets_visitante == 3: return 0.0
    fl = 3 - sets_local
    fv = 3 - sets_visitante
    p, q, prob = prob_set_local, 1 - prob_set_local, 0.0
    for wins in range(fl, fl + fv):
        n, k = wins + fv - 1, wins - 1
        if k < 0 or k > n: continue
        prob += comb(n, k) * (p**k) * (q**(n-k)) * p
    return min(max(prob, 0.0), 1.0)


def barra_prob(prob: float, ancho: int = 24) -> str:
    llenos = round(prob * ancho)
    return "█" * llenos + "░" * (ancho - llenos)


def ajustar_marcador_voley(pts_ganador: int, pts_perdedor: int, es_desempate: bool) -> tuple:
    """
    Ajusta un marcador crudo para que cumpla las reglas del voleibol:
      - Sets 1-4: el ganador llega a 25 mín., con ventaja de 2 mín.
      - Set 5:    el ganador llega a 15 mín., con ventaja de 2 mín.
      - Prórroga: si el perdedor llega al límite-1, se alarga de 2 en 2.
    Devuelve (pts_ganador_corregido, pts_perdedor_corregido).
    """
    minimo = 15 if es_desempate else 25

    # El ganador debe tener al menos `minimo` puntos
    pts_ganador = max(pts_ganador, minimo)

    # Si el perdedor también llegó al límite-1 → prórroga
    if pts_perdedor >= minimo - 1:
        pts_perdedor = max(pts_perdedor, minimo - 1)
        pts_ganador  = pts_perdedor + 2
    else:
        # Sin prórroga: ventaja mínima de 2
        pts_perdedor = min(pts_perdedor, pts_ganador - 2)
        pts_perdedor = max(pts_perdedor, 0)

    return pts_ganador, pts_perdedor


def corregir_marcador_set(ptl_raw: int, ptv_raw: int, gana_local: bool, set_num: int) -> tuple:
    """Aplica las reglas de voleibol al marcador de un set."""
    es_desempate = (set_num == 5)
    if gana_local:
        ptl, ptv = ajustar_marcador_voley(ptl_raw, ptv_raw, es_desempate)
    else:
        ptv, ptl = ajustar_marcador_voley(ptv_raw, ptl_raw, es_desempate)
    return ptl, ptv


def calcular_rango_voley(ptl_est: int, ptv_est: int, gana_local: bool,
                         set_num: int, margen: int = 4) -> tuple:
    """
    Genera el rango (min, max) para local y visitante respetando las reglas.

    El rango del perdedor se DERIVA del rango del ganador aplicando las mismas
    reglas de voleibol, en lugar de calcularse de forma independiente.
    Esto garantiza que en ningún extremo del rango el perdedor >= ganador - 1.

    Devuelve ((lmin, lmax), (vmin, vmax)).
    """
    es_desempate = (set_num == 5)
    minimo = 15 if es_desempate else 25

    if gana_local:
        gan_est, per_est = ptl_est, ptv_est
    else:
        gan_est, per_est = ptv_est, ptl_est

    # Rango del ganador: ±margen desde la estimación central
    gan_min = max(gan_est - margen, minimo)
    gan_max = gan_est + margen

    # Perdedor máximo → partido más igualado (ganador en su mínimo)
    _, per_max = ajustar_marcador_voley(gan_min, per_est + margen, es_desempate)
    # Perdedor mínimo → partido más dominante (ganador en su máximo)
    _, per_min = ajustar_marcador_voley(gan_max, per_est - margen, es_desempate)
    per_min = max(per_min, 0)

    if gana_local:
        return (gan_min, gan_max), (per_min, per_max)
    else:
        return (per_min, per_max), (gan_min, gan_max)


# ═════════════════════════════════════════════════════════════════
# SIMULACIÓN SET A SET
# ═════════════════════════════════════════════════════════════════

def simular_partido(vb: VBPredictor, home: str, away: str, season: str) -> None:
    home = normalize_club(home, season)
    away = normalize_club(away, season)
    W    = 60

    # ── Predicción global ──────────────────────────────────────────
    match_res = vb.match_model.predict(home, away, season)
    hs = match_res["home_sets_rounded"]
    vs = match_res["away_sets_rounded"]
    hc = match_res["home_sets"]
    vc = match_res["away_sets"]

    # Asegurar resultado válido (uno debe tener 3 sets)
    if hs == vs or (hs < 3 and vs < 3):
        if hc >= vc:
            hs, vs = 3, max(0, min(2, round(vc)))
        else:
            vs, hs = 3, max(0, min(2, round(hc)))

    total_sets = hs + vs
    ganador_global = home if hs > vs else away

    print(f"\n{'═' * W}")
    print(f"  {home}  vs  {away}".center(W))
    print(f"  Temporada: {season}  |  Modelo: hasta {MODELO_HASTA.replace('_','/')}".center(W))
    print(f"{'═' * W}")
    print(f"\n  RESULTADO GLOBAL PREDICHO")
    print(f"  {'─' * (W-2)}")
    print(f"  {home:<28} {hs} sets  (continuo: {hc})")
    print(f"  {away:<28} {vs} sets  (continuo: {vc})")
    print(f"  → Ganador: {ganador_global}")

    # ── Verificar que el modelo de sets está entrenado ─────────────
    has_set_model = (
        hasattr(vb, "set_model") and
        vb.set_model is not None and
        getattr(vb.set_model, "feat_cols", None) is not None
    )
    if not has_set_model:
        print(f"\n  ⚠️  Modelo de sets no disponible para este corte.")
        print(f"     Reentrena con trainer.py incluyendo DB/sets_partidos.csv")
        return

    # ── Simulación set a set ───────────────────────────────────────
    sets_local = sets_visit = 0
    print(f"\n  SIMULACIÓN SET A SET")
    print(f"  {'─' * (W-2)}")

    for set_num in range(1, total_sets + 1):
        pred = vb.set_model.predict_set(
            local=home, visitante=away, season=season,
            set_num=set_num,
            sets_local_antes=sets_local,
            sets_visitante_antes=sets_visit,
        )

        pl  = pred["prob_local"]
        pv  = pred["prob_visitante"]
        gl  = pred["gana_local"]

        # ── Aplicar reglas de voleibol al marcador estimado ──────────
        ptl_raw = pred["pts_local_est"]
        ptv_raw = pred["pts_visit_est"]
        ptl, ptv = corregir_marcador_set(ptl_raw, ptv_raw, gl, set_num)

        # ── Calcular rango respetando reglas de voleibol ─────────────
        (lmin, lmax), (vmin, vmax) = calcular_rango_voley(ptl, ptv, gl, set_num)

        prob_antes = prob_partido(pl, sets_local, sets_visit)
        if gl: sets_local  += 1
        else:  sets_visit  += 1
        prob_desp = prob_partido(pl, sets_local, sets_visit)

        delta = prob_desp - prob_antes
        tend  = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
        ganador_set = home if gl else away

        # Formatear marcador y rango con ganador primero
        if gl:
            marc_est  = f"{ptl}-{ptv}"
            rango_str = f"{lmin}-{vmin}  a  {lmax}-{vmax}"
        else:
            marc_est  = f"{ptv}-{ptl}"
            rango_str = f"{vmin}-{lmin}  a  {vmax}-{lmax}"

        print(f"\n  SET {set_num}  [{sets_local - (1 if gl else 0)}-{sets_visit - (0 if gl else 1)} antes]")
        print(f"  {'─' * (W-2)}")
        print(f"  Ganador predicho : {ganador_set}")
        print(f"  Marcador est.    : {marc_est}  (rango: {rango_str})")
        print(f"  {home[:22]:<22} {pl*100:>5.1f}%  {barra_prob(pl)}")
        print(f"  {away[:22]:<22} {pv*100:>5.1f}%  {barra_prob(pv)}")
        print(f"  Prob. partido → {home}: {prob_desp*100:.1f}%  "
              f"{tend}  (antes: {prob_antes*100:.1f}%)")
        print(f"  Marcador parcial : {home} {sets_local} - {sets_visit} {away}")

    ganador_final = home if sets_local > sets_visit else away
    print(f"\n{'═' * W}")
    print(f"  RESULTADO FINAL".center(W))
    print(f"  {home} {sets_local}  —  {sets_visit} {away}".center(W))
    print(f"  Ganador: {ganador_final}".center(W))
    print(f"{'═' * W}\n")


# ═════════════════════════════════════════════════════════════════
# ENTRADA PRINCIPAL
# ═════════════════════════════════════════════════════════════════

def listar_modelos(model_dir: str) -> None:
    modelos = sorted(glob.glob(os.path.join(model_dir, "modelo_hasta_*.pkl")))
    if not modelos:
        print("⚠️  No hay modelos en model/. Ejecuta trainer.py primero.")
    else:
        print("Modelos disponibles:")
        for m in modelos:
            print(f"  {os.path.basename(m)}  ({os.path.getsize(m)//1024} KB)")


if __name__ == "__main__":
    print("=" * 60)
    print("  PREDICTOR VB — Simulación set a set")
    print("=" * 60)

    listar_modelos("model")

    model_path = os.path.join("model", f"modelo_hasta_{MODELO_HASTA}.pkl")
    vb = VBPredictor.load(model_path)

    # Jugadores clave antes del partido
    print(f"\n{'─' * 60}")
    print(f"  JUGADORES DESTACADOS — temporada {SEASON}")
    print(f"{'─' * 60}")
    mostrar_jugadores_destacados(vb, HOME_CLUB, AWAY_CLUB, SEASON)

    # Simulación del partido
    simular_partido(vb, HOME_CLUB, AWAY_CLUB, SEASON)