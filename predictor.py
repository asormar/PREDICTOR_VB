"""
predecir.py — Predice el resultado de un partido sin reentrenar el modelo.

Uso:
    python predecir.py
    python predecir.py --local Perugia --visitante Trento --temporada 2024/2025
    python predecir.py --local Modena --visitante Milano --temporada 2025/2026

Requisito: haber ejecutado prototipo1.py al menos una vez para generar
           el archivo model/predictor_vb.pkl
"""

import argparse
from prototipo1 import SetPredictor, normalize_club

MODEL_PATH = "model/predictor_vb.pkl"

EQUIPOS_DISPONIBLES = [
    "Perugia", "Trento", "Milano", "Modena", "Monza", "Lube",
    "Piacenza", "Verona", "Padova", "Cisterna", "Grottazzolina",
    "Taranto", "Ravenna", "Cuneo", "Siena", "Vibo Valentia",
    "Acicastello",
]


def predecir(local: str, visitante: str, temporada: str) -> None:
    model = SetPredictor.load(MODEL_PATH)

    local_norm    = normalize_club(local)
    visitante_norm = normalize_club(visitante)

    print(f"\n{'─'*45}")
    print(f"  {local_norm:20s} vs  {visitante_norm}")
    print(f"  Temporada: {temporada}")
    print(f"{'─'*45}")

    try:
        result = model.predict(local_norm, visitante_norm, temporada)
    except Exception as e:
        print(f"❌ Error al predecir: {e}")
        return

    hs = result["home_sets_rounded"]
    as_ = result["away_sets_rounded"]
    hf = result["home_sets"]
    af = result["away_sets"]

    # Determinar ganador
    if hs > as_:
        winner = f"🏆 Gana {local_norm}"
    elif as_ > hs:
        winner = f"🏆 Gana {visitante_norm}"
    else:
        winner = "⚖️  Empate técnico (resultado inusual en voleibol)"

    print(f"  {local_norm:20s}  {hs} sets  (raw: {hf})")
    print(f"  {visitante_norm:20s}  {as_} sets  (raw: {af})")
    print(f"\n  {winner}")
    print(f"{'─'*45}\n")


def main():
    parser = argparse.ArgumentParser(description="Predictor de partidos de voleibol")
    parser.add_argument("--local",      type=str, default=None, help="Equipo local")
    parser.add_argument("--visitante",  type=str, default=None, help="Equipo visitante")
    parser.add_argument("--temporada",  type=str, default="2025/2026", help="Temporada (ej: 2024/2025)")
    args = parser.parse_args()

    # Si no se pasan argumentos, modo interactivo
    if args.local is None or args.visitante is None:
        print("\n🏐 PREDICTOR DE VOLEIBOL — Modo interactivo")
        print("─" * 45)
        print("Equipos disponibles:")
        for i, eq in enumerate(EQUIPOS_DISPONIBLES, 1):
            print(f"  {i:2d}. {eq}")
        print()
        local     = input("Equipo LOCAL    : ").strip()
        visitante = input("Equipo VISITANTE: ").strip()
        temporada = input("Temporada [2025/2026]: ").strip() or "2025/2026"
    else:
        local     = args.local
        visitante = args.visitante
        temporada = args.temporada

    predecir(local, visitante, temporada)


if __name__ == "__main__":
    main()