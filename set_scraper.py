"""
scraper_legavolley.py
─────────────────────
Extrae el historial de partidos de la SuperLega y A2 desde legavolley.it
usando la página de enfrentamientos directos (/confronti/).

Estrategia: para cada par de equipos (A, B) carga:
  https://www.legavolley.it/confronti/?Fase=1&IdCasa=A&IdFuori=B&Serie=1&lang=en

Y extrae la tabla "Last direct Matches" con el resultado por set de cada partido.

Salida: DB/sets_partidos.csv (mismo formato que sets_entrenamiento_ejemplo.xlsx)

Uso:
    pip install requests beautifulsoup4
    python scraper_legavolley.py              # Ambas ligas
    python scraper_legavolley.py --liga A1    # Solo SuperLega
    python scraper_legavolley.py --liga A2    # Solo A2
    python scraper_legavolley.py --output DB/mi_archivo.csv

    # Si el HTML tiene estructura distinta, ejecuta primero:
    python scraper_legavolley.py --diagnosticar 6704 6709
"""

import re
import os
import time
import argparse
import logging
import itertools
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# EQUIPOS Y SUS IDs (extraídos del dropdown de legavolley.it)
# ═════════════════════════════════════════════════════════════════

EQUIPOS_A1 = {
    6704: "Allianz Milano",
    6705: "Cisterna Volley",
    6746: "Cucine Lube Civitanova",
    6706: "Gas Sales Bluenergy Piacenza",
    6707: "Itas Trentino",
    6741: "MA Acqua S.Bernardo Cuneo",
    6709: "Rana Verona",
    6710: "Sir Susa Scai Perugia",
    6711: "Sonepar Padova",
    6712: "Valsa Group Modena",
    6708: "Vero Volley Monza",
    6713: "Yuasa Battery Grottazzolina",
}

EQUIPOS_A2 = {
    6714: "Abba Pineto",
    6718: "Alva Inox 2 Emme Service Porto Viro",
    6715: "Banca Macerata Fisiomed MC",
    6716: "Campi Reali Cantù",
    6717: "Consar Ravenna",
    6719: "Emma Villas Codyeco Lupi Siena",
    6722: "Essence Hotels Fano",
    6721: "Gruppo Consoli Sferc Brescia",
    6740: "Prisma La Cascina Taranto",
    6743: "Rinascita Lagonegro",
    6742: "Romeo Sorrento",
    6752: "Sviluppo Sud Catania",
    6723: "Tinet Prata di Pordenone",
    6720: "Virtus Aversa",
}

# Nombres completos (Legavolley) → nombres cortos (compatibles con df_teams)
NOMBRE_CORTO = {
    "Allianz Milano":                      "Milano",
    "Powervolley Milano":                  "Milano",
    "Revivre Milano":                      "Milano",
    "Revivre Axopower Milano":             "Milano",
    "Cisterna Volley":                     "Cisterna",
    "Cisterna Top Volley":                 "Cisterna Top Volley",
    "Cucine Lube Civitanova":              "Lube",
    "Lube Civitanova":                     "Lube",
    "Gas Sales Bluenergy Piacenza":        "Piacenza",
    "Gas Sales Piacenza":                  "Piacenza",
    "Piacenza Copra":                      "Piacenza",
    "Itas Trentino":                       "Trento",
    "Trentino Volley":                     "Trento",
    "MA Acqua S.Bernardo Cuneo":           "Cuneo",
    "Cuneo Volley":                        "Cuneo",
    "Rana Verona":                         "Verona",
    "WithU Verona":                        "Verona",
    "Verona Volley":                       "Verona",
    "NBV Verona":                          "Verona",
    "Calzedonia Verona":                   "Verona",
    "Sir Susa Scai Perugia":               "Perugia",
    "Sir Safety Perugia":                  "Perugia",
    "Sonepar Padova":                      "Padova",
    "Pallavolo Padova":                    "Padova",
    "Valsa Group Modena":                  "Modena",
    "Leo Shoes Modena":                    "Modena",
    "Vero Volley Monza":                   "Monza",
    "Yuasa Battery Grottazzolina":         "Grottazzolina",
    "Consar Ravenna":                      "Ravenna",
    "Emma Villas Codyeco Lupi Siena":      "Siena",
    "Prisma La Cascina Taranto":           "Taranto",
    "Sviluppo Sud Catania":                "Acicastello",
    "Acicastello":                         "Acicastello",
    "Saturnia Acicastello":               "Acicastello",
    "Campi Reali Cantù":                   "Cantù",
    "Gruppo Consoli Sferc Brescia":        "Brescia",
    "Tinet Prata di Pordenone":            "Prata di Pordenone",
    "Abba Pineto":                         "Pineto",
    "Rinascita Lagonegro":                 "Lagonegro",
    "Romeo Sorrento":                      "Sorrento",
    "Virtus Aversa":                       "Aversa",
    "Alva Inox 2 Emme Service Porto Viro": "Porto Viro",
    "Banca Macerata Fisiomed MC":          "Macerata",
    "Essence Hotels Fano":                 "Fano",
}

BASE_URL  = "https://www.legavolley.it"
CONFRONTI = (
    BASE_URL
    + "/confronti/?Fase=1&IdCasa={id_casa}&IdFuori={id_fuori}&Serie={serie}&lang=en"
)
SERIE_A1 = 1
SERIE_A2 = 3

MIN_SEASON_YEAR = 2016   # Descartar partidos anteriores a 2016/2017
DELAY = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,it;q=0.8",
    "Referer": BASE_URL,
}


# ═════════════════════════════════════════════════════════════════
# PARSEO DEL HTML
# ═════════════════════════════════════════════════════════════════

def parse_set_score(score_str: str):
    """'25-22' → (25, 22). '0-0' o vacío → None."""
    if not score_str or score_str.strip() in ("0-0", "-", "", "0"):
        return None
    m = re.match(r"(\d+)-(\d+)", score_str.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_season(season_str: str) -> str:
    """'2025' → '2025/2026'. '2024/2025' → '2024/2025'."""
    s = season_str.strip()
    if "/" in s:
        return s
    try:
        y = int(s)
        return f"{y}/{y + 1}"
    except ValueError:
        return s


def normalize_name(raw: str) -> str:
    s = raw.strip()
    return NOMBRE_CORTO.get(s, s)


def scrape_confronti(
    session: requests.Session,
    id_casa: int,
    id_fuori: int,
    nombre_casa: str,
    nombre_fuori: str,
    serie: int,
) -> list:
    """
    Descarga y parsea la tabla 'Last direct Matches' entre dos equipos.
    Devuelve lista de dicts (una fila por set jugado).

    Estructura de columnas esperada en la tabla HTML:
      [0] Home  [1] Host  [2] FinalResult
      [3] 1°Set [4] 2°Set [5] 3°Set [6] 4°Set [7] 5°Set
      [8] Round [9] Step  [10] Championship  [11] Season
    """
    url = CONFRONTI.format(id_casa=id_casa, id_fuori=id_fuori, serie=serie)

    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"    Error HTTP: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")

    # Diagnóstico confirmó: la tabla correcta tiene cabeceras Home/Host + columnas de Set.
    # Puede haber varias tablas en la página; buscamos la específica de enfrentamientos directos.
    todas = soup.select("table")
    tabla = None
    for t in todas:
        cabeceras = [th.get_text(strip=True) for th in t.select("th")]
        if "Home" in cabeceras and "Host" in cabeceras and any("Set" in c for c in cabeceras):
            tabla = t
            break

    if not tabla:
        log.warning(f"    Sin tabla Home/Host/Set para {nombre_casa} vs {nombre_fuori}")
        return []

    filas = tabla.select("tbody tr") or tabla.select("tr")[1:]
    rows  = []

    for i, tr in enumerate(filas):
        celdas = [td.get_text(strip=True) for td in tr.select("td")]
        if len(celdas) < 8:
            continue

        # Columnas confirmadas por diagnóstico:
        # [0] Home  [1] Host  [2] FinalResult
        # [3] 1°Set [4] 2°Set [5] 3°Set [6] 4°Set [7] 5°Set
        # [8] Round [9] Step  [10] Championship  [11] Season
        local_raw     = celdas[0]
        visitante_raw = celdas[1]
        resultado     = celdas[2]
        sets_raw      = celdas[3:8]
        jornada       = celdas[8]  if len(celdas) > 8  else ""
        fase          = celdas[9]  if len(celdas) > 9  else "Regular Season"
        championship  = celdas[10] if len(celdas) > 10 else ""
        season_raw    = celdas[11] if len(celdas) > 11 else ""

        local     = normalize_name(local_raw)
        visitante = normalize_name(visitante_raw)
        temporada = parse_season(season_raw)

        # Filtrar temporadas anteriores al mínimo configurado
        try:
            if int(temporada.split("/")[0]) < MIN_SEASON_YEAR:
                continue
        except (ValueError, IndexError):
            pass

        # ID único de partido: temporada + equipos ordenados + jornada
        # Ordenamos los equipos para que A-vs-B y B-vs-A generen el mismo ID
        eq_sorted  = "_".join(sorted([local[:5], visitante[:5]]))
        partido_id = f"{temporada}_{eq_sorted}_{jornada.replace(' ', '')}"

        for set_num, score_str in enumerate(sets_raw, start=1):
            parsed = parse_set_score(score_str)
            if parsed is None:
                continue
            pl, pv    = parsed
            ganador   = 1 if pl > pv else 0
            limite    = 15 if set_num == 5 else 25
            set_largo = 1 if max(pl, pv) > limite else 0

            rows.append({
                "partido_id":        partido_id,
                "temporada":         temporada,
                "jornada":           jornada,
                "fase":              fase,
                "championship":      championship,
                "equipo_local":      local,
                "equipo_visitante":  visitante,
                "resultado_final":   resultado,
                "set_num":           set_num,
                "puntos_local":      pl,
                "puntos_visitante":  pv,
                "ganador_set_local": ganador,
                "set_largo":         set_largo,
            })

    return rows


# ═════════════════════════════════════════════════════════════════
# DIAGNÓSTICO — imprime estructura HTML para ajustar selectores
# ═════════════════════════════════════════════════════════════════

def diagnosticar(id_casa: int, id_fuori: int, serie: int = SERIE_A1):
    """
    Descarga la página y muestra tablas + divs relevantes.
    Uso: python scraper_legavolley.py --diagnosticar 6704 6709
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    url = CONFRONTI.format(id_casa=id_casa, id_fuori=id_fuori, serie=serie)
    print(f"\nDiagnosticando: {url}\n")

    r   = session.get(url, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    print("── Tablas ──")
    for i, t in enumerate(soup.select("table")):
        cabeceras = [th.get_text(strip=True) for th in t.select("th")]
        filas     = t.select("tr")
        print(f"  Tabla {i}: {len(filas)} filas | cabeceras: {cabeceras[:10]}")
        data_rows = t.select("tbody tr") or filas[1:]
        if data_rows:
            primera = [td.get_text(strip=True) for td in data_rows[0].select("td")]
            print(f"    Primera fila: {primera}")

    print("\n── Divs con clases clave ──")
    for cls in ["match", "confronti", "last", "result", "partita", "calendario"]:
        els = soup.select(f"[class*='{cls}']")
        if els:
            print(f"  [{cls}]: {len(els)} elementos — clases: {els[0].get('class')}")
            print(f"    Texto: {els[0].get_text(strip=True)[:80]}")

    print(f"\n── HTML (primeros 3000 chars) ──")
    print(r.text[:3000])


# ═════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════

def run(equipos: dict, serie: int, output_path: str, delay: float) -> list:
    session  = requests.Session()
    session.headers.update(HEADERS)

    ids    = list(equipos.keys())
    pares  = list(itertools.permutations(ids, 2))
    total  = len(pares)
    all_rows = []
    vistos   = set()

    log.info(f"Equipos: {len(ids)}  |  Pares a consultar: {total}")

    for n, (id_casa, id_fuori) in enumerate(pares, start=1):
        nombre_casa  = equipos[id_casa]
        nombre_fuori = equipos[id_fuori]
        log.info(f"[{n:>3}/{total}]  {nombre_casa}  vs  {nombre_fuori}")

        rows = scrape_confronti(
            session, id_casa, id_fuori,
            nombre_casa, nombre_fuori, serie,
        )

        # Evitar duplicados: el mismo partido aparece en A-vs-B y en B-vs-A
        nuevos = [r for r in rows if r["partido_id"] not in vistos]
        for r in nuevos:
            vistos.add(r["partido_id"])
        all_rows.extend(nuevos)

        log.info(f"         → {len(nuevos)} sets nuevos  (total: {len(all_rows)})")

        if n % 20 == 0:
            _guardar_progreso(all_rows, output_path)

        time.sleep(delay)

    return all_rows


def _guardar_progreso(rows: list, output_path: str):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(output_path + ".partial", index=False)
    log.info(f"  ▶ Progreso: {len(rows)} sets guardados en {output_path}.partial")


def guardar_final(rows: list, output_path: str):
    if not rows:
        log.error("No se obtuvieron datos. Ejecuta con --diagnosticar para inspeccionar el HTML.")
        return

    df = (
        pd.DataFrame(rows)
        .sort_values(["temporada", "equipo_local", "partido_id", "set_num"])
        .reset_index(drop=True)
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    partial = output_path + ".partial"
    if os.path.exists(partial):
        os.remove(partial)

    print(f"\n{'═'*58}")
    print(f"  SCRAPING COMPLETADO")
    print(f"{'═'*58}")
    print(f"  Archivo  : {output_path}")
    print(f"  Partidos : {df['partido_id'].nunique()}")
    print(f"  Sets     : {len(df)}")
    print(f"\n  Desglose por temporada:")
    for temp, g in df.groupby("temporada"):
        print(f"    {temp}: {g['partido_id'].nunique()} partidos, {len(g)} sets")
    print(f"{'═'*58}\n")


# ═════════════════════════════════════════════════════════════════
# ENTRADA
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scraper de enfrentamientos directos — Legavolley.it"
    )
    parser.add_argument(
        "--liga", choices=["A1", "A2", "ambas"], default="ambas",
        help="Liga a scrapear (default: ambas)",
    )
    parser.add_argument(
        "--output", default="DB/sets_partidos.csv",
        help="Ruta del CSV de salida",
    )
    parser.add_argument(
        "--delay", type=float, default=DELAY,
        help=f"Segundos entre peticiones (default: {DELAY})",
    )
    parser.add_argument(
        "--diagnosticar", nargs=2, type=int, metavar=("ID_CASA", "ID_FUORI"),
        help="Imprime el HTML de un par para ajustar selectores. Ej: --diagnosticar 6704 6709",
    )
    args = parser.parse_args()

    if args.diagnosticar:
        diagnosticar(args.diagnosticar[0], args.diagnosticar[1])
    else:
        print(f"{'═'*58}")
        print(f"  SCRAPER LEGAVOLLEY — Enfrentamientos directos")
        print(f"{'═'*58}")

        all_rows = []

        if args.liga in ("A1", "ambas"):
            print(f"\n  ── SuperLega (A1): {len(EQUIPOS_A1)} equipos ──")
            all_rows += run(EQUIPOS_A1, SERIE_A1, args.output, args.delay)

        if args.liga in ("A2", "ambas"):
            print(f"\n  ── Serie A2: {len(EQUIPOS_A2)} equipos ──")
            all_rows += run(EQUIPOS_A2, SERIE_A2, args.output, args.delay)

        guardar_final(all_rows, args.output)