"""
predictor.py
────────────
Carga el modelo transformer entrenado con trainer.py y simula un partido:

  1. Jugadores clave de cada equipo antes del partido
  2. Módulo 1 — resultado global predicho (con probabilidades por resultado)
  3. Módulo 2 — simulación set a set con probabilidad de ganar cada set
  4. Jugadores clave de cada equipo (resumen al final)

Edita la sección CONFIGURACIÓN y ejecuta:
    python predictor.py
"""

import os
import sys
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ── Importar arquitecturas desde trainer.py ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainer import (
    VBTransformerPredictor, MatchTransformer, SetTransformer,
    PositionalEncoding, TeamEncoder,
    normalize_club, TOP8, SEASONS, MATCH_RESULTS,
    CONTEXT_LEN, TOP_PLAYERS, PLAYER_STAT_COLS,
    compute_team_history,
)


# ═════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════

MODEL_PATH = "model/vb_transformer.pkl"

HOME_CLUB = "Milano"
AWAY_CLUB = "Monza"
SEASON    = "2024/2025"


# ═════════════════════════════════════════════════════════════════
# CORRECCIÓN DE MARCADORES (reglas del voleibol)
# ═════════════════════════════════════════════════════════════════

def ajustar_marcador(pts_gan: int, pts_per: int, es_desempate: bool) -> tuple:
    minimo   = 15 if es_desempate else 25
    pts_gan  = max(pts_gan, minimo)
    if pts_per >= minimo - 1:
        pts_per = max(pts_per, minimo - 1)
        pts_gan = pts_per + 2
    else:
        pts_per = min(pts_per, pts_gan - 2)
        pts_per = max(pts_per, 0)
    return pts_gan, pts_per


def marcador_estimado(prob_local: float, set_num: int) -> tuple:
    """
    Estima un marcador plausible basándose en la probabilidad de victoria
    del local y si es set de desempate.
    """
    es_desemp = (set_num == 5)
    minimo    = 15 if es_desemp else 25
    margen    = 4

    if prob_local >= 0.5:
        # Gana local
        pts_gan = minimo + round((prob_local - 0.5) * 2 * margen)
        pts_per = pts_gan - 2 - round((prob_local - 0.5) * 4)
        pts_per = max(pts_per, 0)
        ptl, ptv = ajustar_marcador(pts_gan, pts_per, es_desemp)
    else:
        # Gana visitante
        pts_gan = minimo + round((0.5 - prob_local) * 2 * margen)
        pts_per = pts_gan - 2 - round((0.5 - prob_local) * 4)
        pts_per = max(pts_per, 0)
        ptv, ptl = ajustar_marcador(pts_gan, pts_per, es_desemp)

    # Rango ±3
    if prob_local >= 0.5:
        lmin, lmax = max(minimo, ptl - 3), ptl + 3
        vmin, vmax = max(0, ptv - 3), min(ptv + 3, lmin - 2)
    else:
        vmin, vmax = max(minimo, ptv - 3), ptv + 3
        lmin, lmax = max(0, ptl - 3), min(ptl + 3, vmin - 2)

    return ptl, ptv, lmin, lmax, vmin, vmax


# ═════════════════════════════════════════════════════════════════
# PROBABILIDAD DE PARTIDO (Monte Carlo)
# ═════════════════════════════════════════════════════════════════

def actualizar_prob_partido(prob_match, prob_set, gana_local):
    """
    Actualiza suavemente la probabilidad global del partido
    según el resultado del set.
    """

    impacto = abs(prob_set - 0.5) * 0.35

    if gana_local:
        prob_match += impacto * (1.0 - prob_match)
    else:
        prob_match -= impacto * prob_match

    return float(np.clip(prob_match, 0.01, 0.99))


# ═════════════════════════════════════════════════════════════════
# JUGADORES DESTACADOS
# ═════════════════════════════════════════════════════════════════

def mostrar_jugadores(vb: VBTransformerPredictor, home: str,
                       away: str, season: str, top_n: int = 5):
    if not vb.player_feats:
        print("  (datos de jugadores no disponibles)")
        return

    stat_labels = {
        "pts_set": "Puntos/set",
        "att_set": "Ataques ganados/set",
        "ace_set": "Aces/set",
        "rec_set": "Recepciones exc./set",
        "blk_set": "Bloqueos ganados/set",
    }
    stat_keys = list(PLAYER_STAT_COLS.keys())

    # Cargar datos raw para mostrar jugadores individuales
    player_files = {
        "pts_set": "DB/stats_jugadores_set/Points_Set_filtered.xlsx",
        "att_set": "DB/stats_jugadores_set/Won_Attacks_Set_filtered.xlsx",
        "ace_set": "DB/stats_jugadores_set/Ace_Set_filtered.xlsx",
        "rec_set": "DB/stats_jugadores_set/Excellent_Receptions_Set_filtered.xlsx",
        "blk_set": "DB/stats_jugadores_set/Won_Blocks_Set_filtered.xlsx",
    }
    stat_cols = {k: v for k, v in PLAYER_STAT_COLS.items()}

    frames = []
    for key, path in player_files.items():
        if not os.path.exists(path): continue
        col = stat_cols[key]
        df  = pd.read_excel(path)
        df["team_norm"] = df["Team"].apply(normalize_club)
        df["season"]    = df["Season"].astype(str)
        tmp = df[["Player","team_norm","season","Played Sets", col]].copy()
        tmp.columns = ["player","team","season","sets_played","stat_val"]
        tmp["stat"] = key
        tmp["sets_played"] = pd.to_numeric(tmp["sets_played"], errors="coerce").fillna(1)
        tmp["stat_val"]    = pd.to_numeric(tmp["stat_val"],    errors="coerce").fillna(0)
        frames.append(tmp)

    if not frames:
        print("  (archivos de jugadores no encontrados)")
        return

    all_stats = pd.concat(frames, ignore_index=True)

    for team_name in [home, away]:
        team_norm = normalize_club(team_name)
        print(f"\n  ── {team_name} ──")
        df_t = all_stats[
            (all_stats["team"] == team_norm) &
            (all_stats["season"] == season)
        ]
        if df_t.empty:
            print(f"    (sin datos para {season})")
            continue

        # Calcular score global por jugador
        rows = []
        for (player, stat), g in df_t.groupby(["player","stat"]):
            w = g["sets_played"].values
            v = g["stat_val"].values
            rows.append({"player": player, "stat": stat,
                         "val": np.average(v, weights=w) if w.sum() > 0 else 0})

        if not rows: continue
        pivot = pd.DataFrame(rows).pivot(
            index="player", columns="stat", values="val"
        ).fillna(0).reindex(columns=stat_keys, fill_value=0)

        pivot_n = pivot.copy()
        for col in pivot_n.columns:
            mn, mx = pivot_n[col].min(), pivot_n[col].max()
            pivot_n[col] = (pivot_n[col] - mn) / (mx - mn) if mx > mn else 0.0
        pivot["score"] = pivot_n.mean(axis=1)
        top = pivot.sort_values("score", ascending=False).head(top_n)

        for player, row in top.iterrows():
            bar = "█" * int(row["score"] * 20)
            stats_str = "  ".join(
                f"{stat_labels[s]}: {row[s]:.2f}"
                for s in stat_keys if s in row.index
            )
            print(f"    {player:<35} score: {row['score']:.3f}  {bar}")
            print(f"      {stats_str}")


# ═════════════════════════════════════════════════════════════════
# PREDICCIÓN MÓDULO 1 — RESULTADO GLOBAL
# ═════════════════════════════════════════════════════════════════

def predecir_partido(vb: VBTransformerPredictor,
                     home: str, away: str, season: str) -> dict | None:
    if vb.match_model is None:
        print("  ⚠️  MatchTransformer no disponible.")
        return None

    home_n = normalize_club(home)
    away_n = normalize_club(away)

    # Buscar el índice del partido en el histórico para extraer contexto
    hist = vb.match_history
    idx  = len(hist)  # Predecir como si fuera el siguiente partido

    seq_l = compute_team_history(hist, home_n, idx)
    seq_v = compute_team_history(hist, away_n, idx)

    if seq_l is None or seq_v is None:
        print(f"  ⚠️  Historial insuficiente para {home_n} o {away_n}")
        return None

    t_l = torch.tensor(seq_l, dtype=torch.float32).unsqueeze(0)
    t_v = torch.tensor(seq_v, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = vb.match_model(t_l, t_v)
        probs  = torch.softmax(logits, dim=-1).squeeze().numpy()

    # 3 clases: 0=local gana, 1=visitante gana
    pred_idx = int(probs.argmax())
    prob_local  = float(probs[0])
    prob_visit  = float(probs[1])

    return {
        "resultado":  "local" if pred_idx == 0 else "visitante",
        "prob_local": prob_local,
        "prob_visit": prob_visit,
        "probs":      {"local gana": prob_local, "visitante gana": prob_visit},
    }


# ═════════════════════════════════════════════════════════════════
# PREDICCIÓN MÓDULO 2 — SET A SET
# ═════════════════════════════════════════════════════════════════

def predecir_set(vb: VBTransformerPredictor,
                 home: str, away: str, season: str,
                 set_num: int = 1,
                 sets_local_antes: int = 0,
                 sets_visit_antes: int = 0) -> float | None:
    """Devuelve P(gana local el set) usando el SetTransformer."""
    if vb.set_model is None or vb.set_scaler is None:
        return None

    home_n = normalize_club(home)
    away_n = normalize_club(away)

    fl = vb.player_feats.get((home_n, season))
    fv = vb.player_feats.get((away_n, season))

    if fl is None or fv is None:
        return None

    ctx = np.array([
        set_num / 5.0,
        sets_local_antes / 3.0,
        sets_visit_antes / 3.0,
        (sets_local_antes - sets_visit_antes) / 2.0,
    ], dtype=np.float32)
    x = np.concatenate([fl, fv, ctx]).reshape(1, -1)
    x = vb.set_scaler.transform(x)
    t = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        logits = vb.set_model(t)
        probs  = torch.softmax(logits, dim=-1).squeeze().numpy()

    return float(probs[1])   # prob de que gane el local


def simular_sets(
        vb,
        home,
        away,
        season,
        total_sets,
        prob_match_local,
    ):
    W = 60
    sets_local = sets_visit = 0
    print(f"\n  SIMULACIÓN SET A SET")
    print(f"  {'─' * (W - 4)}")
    
    prob_partido_actual = prob_match_local

    for set_num in range(1, total_sets + 1):
        # El partido termina cuando alguien llega a 3 sets
        if sets_local >= 3 or sets_visit >= 3:
            break
        prob_l = predecir_set(
            vb,
            home,
            away,
            season,
            set_num=set_num,
            sets_local_antes=sets_local,
            sets_visit_antes=sets_visit,
        )
        
        if prob_l is None:
            print("  ⚠️  SetTransformer no disponible para este partido.")
            break

        # Combinar predicción del set con la del partido
        prob_l = 0.7 * prob_l + 0.3 * prob_match_local

        # Calibrar probabilidades extremas
        prob_l = calibrar_probabilidad(prob_l)

        prob_v    = 1.0 - prob_l
        gana_l = np.random.random() < prob_l
        ganador   = home if gana_l else away

        ptl, ptv, lmin, lmax, vmin, vmax = marcador_estimado(prob_l, set_num)

        prob_antes = prob_partido_actual

        if gana_l:
            sets_local += 1
        else:
            sets_visit += 1

        prob_partido_actual = actualizar_prob_partido(
            prob_partido_actual,
            prob_l,
            gana_l
        )

        prob_desp = prob_partido_actual

        delta = prob_desp - prob_antes
        tend  = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")

        marc = f"{ptl}-{ptv}" if gana_l else f"{ptv}-{ptl}"
        rng  = (f"{lmin}-{vmin} a {lmax}-{vmax}" if gana_l
                else f"{vmin}-{lmin} a {vmax}-{lmax}")

        bar_l = "█" * int(prob_l * 24) + "░" * (24 - int(prob_l * 24))
        bar_v = "█" * int(prob_v * 24) + "░" * (24 - int(prob_v * 24))

        print(f"\n  SET {set_num}  [{sets_local - (1 if gana_l else 0)}"
              f"-{sets_visit - (0 if gana_l else 1)} antes]")
        print(f"  {'─' * (W - 4)}")
        print(f"  Ganador predicho : {ganador}")
        print(f"  Marcador est.    : {marc}  (rango: {rng})")
        print(f"  {home[:22]:<22} {prob_l*100:>5.1f}%  {bar_l}")
        print(f"  {away[:22]:<22} {prob_v*100:>5.1f}%  {bar_v}")
        print(f"  Prob. partido → {home}: {prob_desp*100:.1f}%  "
              f"{tend}  (antes: {prob_antes*100:.1f}%)")
        print(f"  Parcial: {home} {sets_local} - {sets_visit} {away}")

    ganador_final = home if sets_local > sets_visit else away
    print(f"\n{'═' * W}")
    print(f"  RESULTADO FINAL".center(W))
    print(f"  {home} {sets_local}  —  {sets_visit} {away}".center(W))
    print(f"  Ganador: {ganador_final}".center(W))
    print(f"{'═' * W}\n")

def calibrar_probabilidad(prob: float,
                           strength: float = 0.4,
                           min_p: float = 0.15,
                           max_p: float = 0.85) -> float:
    """
    Reduce probabilidades extremas y mejora calibración.
    """
    # Comprimir hacia 0.5
    prob = 0.5 + (prob - 0.5) * strength

    # Limitar extremos
    prob = np.clip(prob, min_p, max_p)

    return float(prob)


# ═════════════════════════════════════════════════════════════════
# ENTRADA PRINCIPAL
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    W = 60
    print("═" * W)
    print("  PREDICTOR VB — Transformer".center(W))
    print("═" * W)

    if not os.path.exists(MODEL_PATH):
        print(f"\n  ❌ Modelo no encontrado: '{MODEL_PATH}'")
        print("     Ejecuta trainer.py primero.")
        exit(1)

    vb = VBTransformerPredictor.load(MODEL_PATH)
    print(f"  Modelo cargado: {MODEL_PATH}")

    print(f"\n{'─' * W}")
    print(f"  {HOME_CLUB}  vs  {AWAY_CLUB}  |  {SEASON}".center(W))
    print(f"{'─' * W}")

    # 1. Jugadores clave antes del partido
    print(f"\n  JUGADORES DESTACADOS — {SEASON}")
    print(f"  {'─' * (W - 4)}")
    mostrar_jugadores(vb, HOME_CLUB, AWAY_CLUB, SEASON)

    # 2. Resultado global
    print(f"\n{'─' * W}")
    print(f"  MÓDULO 1 — Resultado global predicho")
    print(f"{'─' * W}")
    res = predecir_partido(vb, HOME_CLUB, AWAY_CLUB, SEASON)

    if res:
        ganador = HOME_CLUB if res["resultado"] == "local" else AWAY_CLUB
        perdedor = AWAY_CLUB if res["resultado"] == "local" else HOME_CLUB
        prob_g = res["prob_local"] if res["resultado"] == "local" else res["prob_visit"]
        print(f"\n  Ganador predicho : {ganador}  ({prob_g:.1%})")
        print(f"  {HOME_CLUB:<28} prob. victoria: {res['prob_local']:.1%}")
        print(f"  {AWAY_CLUB:<28} prob. victoria: {res['prob_visit']:.1%}")

        # El número de sets a simular se estima a partir de la confianza:
        # alta confianza (>70%) → probablemente 3-0 o 3-1 (3-4 sets)
        # baja confianza (<60%) → probablemente 3-2 (5 sets)
        prob_max = max(res["prob_local"], res["prob_visit"])
        if prob_max > 0.80:
            total = 3
        elif prob_max > 0.65:
            total = 4
        else:
            total = 5
        print(f"  Sets estimados   : {total}")

        # 3. Simulación set a set
        print(f"\n{'─' * W}")
        print(f"  MÓDULO 2 — Simulación set a set")
        print(f"{'─' * W}")
        simular_sets(
            vb,
            HOME_CLUB,
            AWAY_CLUB,
            SEASON,
            total_sets=total,
            prob_match_local=res["prob_local"],
        )
    else:
        print("  No se pudo generar predicción.")

    # 4. Resumen jugadores al final
    print(f"{'─' * W}")
    print(f"  REFERENCIA RENDIMIENTO TEMPORADA {SEASON}")
    print(f"{'─' * W}")
    mostrar_jugadores(vb, HOME_CLUB, AWAY_CLUB, SEASON)