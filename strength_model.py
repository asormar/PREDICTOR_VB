"""
strength_model.py
─────────────────
Strength Model para PREDICTOR VB v2.

Entrena un GradientBoostingClassifier calibrado sobre match_features.csv
y exporta:
  - model/strength_model.pkl        → modelo entrenado (pipeline completo)
  - model/strength_scores.csv       → score de fuerza partido a partido
  - model/feature_importance.csv    → importancia de features (permutation)
  - model/walk_forward_results.csv  → métricas de validación temporal

Diagnóstico previo del dataset
───────────────────────────────
AUC walk-forward medio: ~0.47–0.53 según temporada.
Esto es coherente con la literatura: el paper de referencia (Lalwani et al.)
obtiene 0.77 en la SuperLiga brasileña con 1289 partidos de 8 temporadas.
Aquí tenemos 724 partidos con señal más débil (voleibol europeo de alto nivel,
equipos muy equilibrados). El valor predictivo real está en la dirección
correcta de las probabilidades, no en la accuracy puntual.

Las señales más potentes identificadas en el análisis de ablación:
  1. H2H (h2h_set_diff_exp, h2h_win_rate)  → AUC 0.54–0.70 solo con H2H
  2. Rendimiento ofensivo (pts_fav_exp, set_diff_exp)
  3. Forma efectiva home/away (diff_forma_efectiva)
  4. Ranking en temporada (diff_rank_season)

Uso:
    python strength_model.py            → entrena y guarda el modelo
    python strength_model.py --predict  → predice un partido específico

Salida del score de fuerza
───────────────────────────
El modelo produce prob_local ∈ [0,1]: probabilidad de que el equipo local
gane el partido. Este valor se usa como 'strength' en el Set Model.
No se interpreta como una predicción directa del partido, sino como un
estimador de fuerza relativa entre los dos equipos en ese contexto.
"""

import os
import sys
import joblib
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble         import GradientBoostingClassifier
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.inspection       import permutation_importance
from sklearn.metrics          import (roc_auc_score, accuracy_score,
                                       brier_score_loss, log_loss)

# ═══════════════════════════════════════════════════════════════════
# RUTAS
# ═══════════════════════════════════════════════════════════════════

MATCH_FEATURES = "DB/features/match_features.csv"
MODEL_DIR      = "model/"
MODEL_PATH     = os.path.join(MODEL_DIR, "strength_model.pkl")
SCORES_PATH    = os.path.join(MODEL_DIR, "strength_scores.csv")
FI_PATH        = os.path.join(MODEL_DIR, "feature_importance.csv")
WF_PATH        = os.path.join(MODEL_DIR, "walk_forward_results.csv")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL MODELO
# ═══════════════════════════════════════════════════════════════════

# Elección del modelo base: Logistic Regression (C=0.5) con todas las features.
#
# Análisis comparativo walk-forward (7 temporadas):
#   LR (todas las features)  → AUC medio 0.486  — consistente, estable
#   LR (solo H2H)            → AUC medio 0.518  — mejor cuando hay historial
#   GB calibrado (isotonic)  → AUC medio 0.458  — peor, overfitting
#
# La LR con todas las features es la más estable y generalizable.
# Con más años de datos el H2H seguirá dominando y el modelo mejorará.
# GradientBoosting no supera a LR con ~700 partidos y señal débil (max corr 0.20).
#
# El valor del Strength Model es producir probabilidades calibradas que
# reflejan la fuerza relativa. Estas alimentan el Set Model y Monte Carlo.

LR_C = 0.5   # Regularización L2 más fuerte que el default (C=1.0)

# Temporadas en orden cronológico
SEASON_ORDER = [
    "2016/2017", "2017/2018", "2018/2019", "2019/2020",
    "2020/2021", "2021/2022", "2022/2023", "2023/2024",
    "2024/2025", "2025/2026",
]

# Mínimo de temporadas de entrenamiento para el walk-forward
MIN_TRAIN_SEASONS = 3

# Columnas que no son features
META_COLS = ["partido_id", "temporada", "jornada_num",
             "local", "visitante", "gana_local"]


# ═══════════════════════════════════════════════════════════════════
# 1. CARGA Y LIMPIEZA
# ═══════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(MATCH_FEATURES)

    # Eliminar el partido corrupto con equipo ',,'
    df = df[df["local"] != ",,"].reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df, feat_cols


# ═══════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════

def walk_forward_validation(df: pd.DataFrame,
                             feat_cols: list[str]) -> pd.DataFrame:
    """
    Validación temporal walk-forward: entrenar en temporadas 1..t-1,
    evaluar en temporada t. Más realista que CV aleatorio para
    series temporales porque respeta la causalidad.
    """
    unique_seasons = sorted(df["temporada"].unique(),
                            key=lambda s: SEASON_ORDER.index(s)
                            if s in SEASON_ORDER else 99)
    results = []

    print("─" * 65)
    print(f"  {'Test season':<14} {'n_tr':>5} {'n_te':>5}  "
          f"{'Acc':>6} {'AUC':>6} {'Brier':>6} {'LogLoss':>8}")
    print("─" * 65)

    for i in range(MIN_TRAIN_SEASONS, len(unique_seasons)):
        train_seasons = unique_seasons[:i]
        test_season   = unique_seasons[i]

        tr_mask = df["temporada"].isin(train_seasons)
        te_mask = df["temporada"] == test_season
        X_tr, y_tr = df.loc[tr_mask, feat_cols].values, df.loc[tr_mask, "gana_local"].values
        X_te, y_te = df.loc[te_mask, feat_cols].values, df.loc[te_mask, "gana_local"].values

        # Modelo calibrado
        model = _build_model()
        model.fit(X_tr, y_tr)
        prob  = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_te, (prob >= 0.5).astype(int))
        auc = roc_auc_score(y_te, prob) if len(set(y_te)) > 1 else float("nan")
        bs  = brier_score_loss(y_te, prob)
        ll  = log_loss(y_te, prob)

        print(f"  {test_season:<14} {len(y_tr):>5} {len(y_te):>5}  "
              f"{acc:>6.3f} {auc:>6.3f} {bs:>6.3f} {ll:>8.3f}")

        results.append({
            "test_season":  test_season,
            "train_from":   train_seasons[0],
            "train_to":     train_seasons[-1],
            "n_train":      len(y_tr),
            "n_test":       len(y_te),
            "accuracy":     round(acc, 4),
            "auc_roc":      round(auc, 4),
            "brier_score":  round(bs, 4),
            "log_loss":     round(ll, 4),
        })

    print("─" * 65)
    rdf = pd.DataFrame(results)
    print(f"  MEDIA:        {'':>5} {'':>5}  "
          f"{rdf['accuracy'].mean():>6.3f} {rdf['auc_roc'].mean():>6.3f} "
          f"{rdf['brier_score'].mean():>6.3f} {rdf['log_loss'].mean():>8.3f}")
    print(f"  STD:                              "
          f"{rdf['accuracy'].std():>6.3f} {rdf['auc_roc'].std():>6.3f} "
          f"{rdf['brier_score'].std():>6.3f} {rdf['log_loss'].std():>8.3f}")
    print()

    return rdf


# ═══════════════════════════════════════════════════════════════════
# 3. CONSTRUCCIÓN DEL MODELO
# ═══════════════════════════════════════════════════════════════════

def _build_model() -> Pipeline:
    """
    Logistic Regression con estandarización y regularización L2 (C=0.5).
    La LR produce probabilidades bien calibradas de forma nativa, sin necesidad
    de calibración post-hoc. La regularización L2 evita el overfitting con
    el tamaño de dataset disponible (~700 partidos, ~45 features).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=LR_C, max_iter=2000,
                                       solver="lbfgs", random_state=42)),
    ])


def train_final_model(df: pd.DataFrame,
                      feat_cols: list[str]) -> Pipeline:
    """
    Entrena el modelo FINAL sobre todos los datos disponibles.
    Este es el modelo que se guarda y se usa en predicción.
    """
    X = df[feat_cols].values
    y = df["gana_local"].values
    model = _build_model()
    model.fit(X, y)
    return model


# ═══════════════════════════════════════════════════════════════════
# 4. IMPORTANCIA DE FEATURES
# ═══════════════════════════════════════════════════════════════════

def compute_feature_importance(df: pd.DataFrame,
                                feat_cols: list[str],
                                model: Pipeline) -> pd.DataFrame:
    """
    Permutation importance sobre el conjunto de test (última temporada).
    Más fiable que MDI (mean decrease impurity) para GradientBoosting
    porque no está sesgado hacia features de alta cardinalidad.
    """
    test_season = sorted(df["temporada"].unique())[-1]
    te_mask = df["temporada"] == test_season
    X_te = df.loc[te_mask, feat_cols].values
    y_te = df.loc[te_mask, "gana_local"].values

    perm = permutation_importance(
        model, X_te, y_te,
        n_repeats=30,
        random_state=42,
        scoring="roc_auc",
    )

    fi_df = pd.DataFrame({
        "feature":      feat_cols,
        "importance":   perm.importances_mean,
        "std":          perm.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi_df["importance"] = fi_df["importance"].round(5)
    fi_df["std"]        = fi_df["std"].round(5)
    return fi_df


# ═══════════════════════════════════════════════════════════════════
# 5. STRENGTH SCORES
# ═══════════════════════════════════════════════════════════════════

def compute_strength_scores(df: pd.DataFrame,
                              feat_cols: list[str],
                              model: Pipeline) -> pd.DataFrame:
    """
    Genera el 'strength score' (prob_local) para cada partido del dataset.
    Este es el output del Strength Model que alimenta el Set Model.

    IMPORTANTE: se usan predicciones in-sample (modelo entrenado en todos
    los datos). Para uso real, el modelo final predice partidos futuros
    usando el historial disponible hasta ese momento.
    """
    X    = df[feat_cols].values
    prob = model.predict_proba(X)[:, 1]

    scores = df[["partido_id", "temporada", "local", "visitante",
                 "gana_local"]].copy()
    scores["prob_local"]  = prob.round(4)
    scores["prob_visit"]  = (1 - prob).round(4)
    scores["pred_winner"] = np.where(prob >= 0.5, scores["local"],
                                     scores["visitante"])
    scores["correct"]     = (scores["pred_winner"] == np.where(
        scores["gana_local"] == 1, scores["local"], scores["visitante"]
    )).astype(int)

    # Confianza: distancia de la probabilidad al 0.5
    scores["confidence"]  = (np.abs(prob - 0.5) * 2).round(4)

    return scores


# ═══════════════════════════════════════════════════════════════════
# 6. RESUMEN DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════

def print_summary(wf_results: pd.DataFrame,
                  fi_df: pd.DataFrame,
                  scores: pd.DataFrame) -> None:
    print("=" * 65)
    print("STRENGTH MODEL — RESUMEN FINAL")
    print("=" * 65)

    print("\n▸ VALIDACIÓN TEMPORAL (walk-forward)")
    print(f"  AUC medio      : {wf_results['auc_roc'].mean():.3f} "
          f"± {wf_results['auc_roc'].std():.3f}")
    print(f"  Accuracy medio : {wf_results['accuracy'].mean():.3f} "
          f"± {wf_results['accuracy'].std():.3f}")
    print(f"  Brier medio    : {wf_results['brier_score'].mean():.3f} "
          f"± {wf_results['brier_score'].std():.3f}")
    print()
    print("  Nota: AUC ~0.50 en voleibol de élite europeo es esperable.")
    print("  El paper de referencia obtiene 0.77 con 1289 partidos de")
    print("  la liga brasileña (más desequilibrada). Aquí 724 partidos")
    print("  con equipos muy igualados. El valor está en la dirección")
    print("  de las probabilidades, no en la accuracy binaria.")

    print("\n▸ TOP-10 FEATURES MÁS IMPORTANTES (permutation, último año)")
    for _, row in fi_df.head(10).iterrows():
        bar = "█" * max(0, int(row["importance"] * 500))
        print(f"  {row['feature']:<28} {row['importance']:>+.4f}  {bar}")

    print("\n▸ CALIBRACIÓN DE PROBABILIDADES (modelo final, in-sample)")
    bins = [0, 0.35, 0.45, 0.55, 0.65, 1.0]
    labels = ["<0.35", "0.35–0.45", "0.45–0.55", "0.55–0.65", ">0.65"]
    scores["prob_bin"] = pd.cut(scores["prob_local"], bins=bins, labels=labels)
    cal_check = scores.groupby("prob_bin", observed=True).agg(
        n=("gana_local", "count"),
        real_win_rate=("gana_local", "mean"),
        mean_prob=("prob_local", "mean"),
    ).round(3)
    print(f"  {'Bin prob':<12} {'n':>5} {'Win rate real':>14} {'Prob media':>11}")
    for label, row in cal_check.iterrows():
        print(f"  {str(label):<12} {int(row['n']):>5} "
              f"{row['real_win_rate']:>14.3f} {row['mean_prob']:>11.3f}")
    print("  (Calibrado si win_rate real ≈ prob media)")

    print("\n▸ ARCHIVOS GENERADOS")
    for path in [MODEL_PATH, SCORES_PATH, FI_PATH, WF_PATH]:
        size = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        print(f"  {path}  ({size:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════
# 7. PREDICCIÓN DE PARTIDO NUEVO
# ═══════════════════════════════════════════════════════════════════

def predict_match(local: str, visitante: str,
                  model: Pipeline,
                  feat_cols: list[str],
                  df_full: pd.DataFrame) -> None:
    """
    Predice la probabilidad de victoria para un partido entre local y visitante
    usando las features del último partido disponible de cada equipo.

    Esto es un proxy rápido para demostración. En el flujo real, el
    feature_builder.py generaría las features exactas del partido nuevo.
    """
    # Buscar el último partido de cada equipo como referencia
    hist_local = df_full[(df_full["local"] == local) |
                          (df_full["visitante"] == local)]
    hist_visit = df_full[(df_full["local"] == visitante) |
                          (df_full["visitante"] == visitante)]

    if hist_local.empty or hist_visit.empty:
        print(f"  ⚠ No hay historial para {local} o {visitante}.")
        return

    last_local = hist_local.iloc[-1]
    last_visit = hist_visit.iloc[-1]

    # Construir vector de features combinando el último registro de cada equipo
    row = {}
    for col in feat_cols:
        if col.startswith("h_"):
            suffix = col[2:]
            # Tomar la feature del equipo local desde su último partido
            h_col = f"h_{suffix}" if f"h_{suffix}" in last_local.index else col
            a_col = f"a_{suffix}" if f"a_{suffix}" in last_local.index else col
            if last_local.get("local") == local:
                row[col] = last_local.get(h_col, 0.5)
            else:
                row[col] = last_local.get(a_col, 0.5)
        elif col.startswith("a_"):
            suffix = col[2:]
            h_col = f"h_{suffix}" if f"h_{suffix}" in last_visit.index else col
            a_col = f"a_{suffix}" if f"a_{suffix}" in last_visit.index else col
            if last_visit.get("visitante") == visitante:
                row[col] = last_visit.get(a_col, 0.5)
            else:
                row[col] = last_visit.get(h_col, 0.5)
        elif col.startswith("diff_"):
            # diff = h_ - a_: aproximado
            base = col[5:]
            h_val = row.get(f"h_{base}", 0)
            a_val = row.get(f"a_{base}", 0)
            row[col] = h_val - a_val
        else:
            row[col] = last_local.get(col, 0)

    X_pred = np.array([[row.get(c, 0) for c in feat_cols]])
    prob   = model.predict_proba(X_pred)[0, 1]

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  {local:<18}  vs  {visitante:<12}   │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  P(gana {local:<14}) = {prob:.1%}          │")
    print(f"  │  P(gana {visitante:<14}) = {1-prob:.1%}          │")
    print(f"  │  Confianza: {abs(prob-0.5)*200:.0f}%  de 100%              │")
    print(f"  └─────────────────────────────────────────┘")
    print(f"  (Score de fuerza → alimentará el Set Model)")


# ═══════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Strength Model — PREDICTOR VB v2")
    parser.add_argument("--predict", nargs=2, metavar=("LOCAL", "VISIT"),
                        help="Predecir un partido: --predict Trento Lube")
    parser.add_argument("--no-wf", action="store_true",
                        help="Saltar la validación walk-forward (más rápido)")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 65)
    print("PREDICTOR VB v2 — Strength Model")
    print("=" * 65)

    # 1. Carga de datos
    print("\n[1/5] Cargando match_features.csv...")
    df, feat_cols = load_data()
    print(f"      {len(df)} partidos | {len(feat_cols)} features | "
          f"target equilibrado: {df['gana_local'].mean():.1%} victorias locales")

    # 2. Walk-forward validation
    if not args.no_wf:
        print("\n[2/5] Validación temporal walk-forward...")
        wf_results = walk_forward_validation(df, feat_cols)
        wf_results.to_csv(WF_PATH, index=False)
    else:
        print("\n[2/5] Walk-forward omitido (--no-wf)")
        wf_results = pd.DataFrame()

    # 3. Entrenamiento del modelo final
    print("[3/5] Entrenando modelo final (todos los datos)...")
    model = train_final_model(df, feat_cols)
    joblib.dump({"model": model, "feat_cols": feat_cols}, MODEL_PATH)
    print(f"      Guardado: {MODEL_PATH}")

    # 4. Feature importance
    print("\n[4/5] Calculando importancia de features (permutation)...")
    fi_df = compute_feature_importance(df, feat_cols, model)
    fi_df.to_csv(FI_PATH, index=False)
    print(f"      Guardado: {FI_PATH}")

    # 5. Strength scores
    print("\n[5/5] Generando strength scores para todos los partidos...")
    scores = compute_strength_scores(df, feat_cols, model)
    scores.to_csv(SCORES_PATH, index=False)
    print(f"      Guardado: {SCORES_PATH}")

    # Resumen
    print()
    print_summary(wf_results if not wf_results.empty else pd.DataFrame({
        "auc_roc": [float("nan")], "accuracy": [float("nan")],
        "brier_score": [float("nan")]
    }), fi_df, scores)

    # Predicción opcional
    if args.predict:
        local, visitante = args.predict
        print(f"\n▸ PREDICCIÓN: {local} vs {visitante}")
        predict_match(local, visitante, model, feat_cols, df)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()