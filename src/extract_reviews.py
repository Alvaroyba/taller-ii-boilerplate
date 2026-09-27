import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://amazon-reviews-api-g5ae.onrender.com"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
REVIEWS_ENDPOINT = f"{BASE_URL}/reviews"
SAMPLE_ENDPOINT = f"{BASE_URL}/reviews/sample"

DEFAULT_OUTPUT_PATH = Path("data/raw/amazon_reviews_raw.csv")
BATCH_SIZE = 1000


def check_server_status(timeout_seconds: int = 180) -> bool:
    """
    Verifica la disponibilidad de la API y maneja el Cold Start de Render.
    Envía peticiones a /health hasta recibir HTTP 200 o agotar el tiempo límite.
    """
    print("Verificando estado del servidor en Render (Cold Start handling)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=15)
            if response.status_code == 200:
                print(f" Servidor activo y respondiendo correctamente (t={int(time.time() - start_time)}s).\n")
                return True
        except requests.RequestException:
            pass

        print(">> Esperando que la instancia en Render despierte (reintento en 5s)...")
        time.sleep(5)

    print(" Error: Se agotó el tiempo esperando que el servidor responda.")
    return False


def extract_reviews_paginated(
    output_path: Path,
    limit_total: int | None = None,
    batch_size: int = BATCH_SIZE,
    resume: bool = True
) -> pd.DataFrame:
    """
    Descarga reseñas de forma paginada desde el endpoint /reviews.
    Soporta reanudación desde archivos parciales existentes y reintentos ante fallos de red.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_reviews = []
    initial_offset = 0

    # Soporte para reanudar descargas interrumpidas
    if resume and output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            initial_offset = len(existing_df)
            if initial_offset > 0:
                print(f"ℹ️ Archivo existente detectado con {initial_offset:,} registros. Reanudando desde offset {initial_offset}...")
                all_reviews = existing_df.to_dict(orient="records")
        except Exception as e:
            print(f"⚠️ No se pudo leer el archivo existente ({e}). Iniciando extracción desde offset 0.")
            initial_offset = 0

    offset = initial_offset
    total_matching = None
    progress_bar = None

    print(f"Iniciando extracción con lotes de {batch_size} registros...")

    while True:
        # Si se especificó un límite total y ya lo alcanzamos
        if limit_total is not None and len(all_reviews) >= limit_total:
            all_reviews = all_reviews[:limit_total]
            break

        current_batch_size = batch_size
        if limit_total is not None:
            remaining = limit_total - len(all_reviews)
            current_batch_size = min(batch_size, remaining)

        params = {"limit": current_batch_size, "offset": offset}

        try:
            res = requests.get(REVIEWS_ENDPOINT, params=params, timeout=30)

            if res.status_code != 200:
                print(f"\n[Aviso] HTTP {res.status_code} en offset {offset}. Reintentando en 5s...")
                time.sleep(5)
                continue

            payload = res.json()
            batch = payload.get("data", [])
            returned = payload.get("returned", len(batch))

            if total_matching is None:
                total_matching = payload.get("total_matching", 210000)
                target_total = min(total_matching, limit_total) if limit_total else total_matching
                progress_bar = tqdm(
                    total=target_total,
                    initial=len(all_reviews),
                    desc="Descargando reseñas",
                    unit="reviews"
                )

            if not batch:
                print("\n[Aviso] No se recibieron más registros de la API.")
                break

            all_reviews.extend(batch)
            if progress_bar:
                progress_bar.update(len(batch))

            offset += returned

            # Guardado incremental de seguridad cada 10.000 registros
            if len(all_reviews) % 10000 == 0:
                temp_df = pd.DataFrame(all_reviews)
                temp_df.to_csv(output_path, index=False)

            if offset >= total_matching:
                break

            # Pequeña pausa para no saturar la conexión
            time.sleep(0.05)

        except requests.RequestException as err:
            print(f"\n[Error de conexión] {err}. Reintentando offset {offset} en 5s...")
            time.sleep(5)

    if progress_bar:
        progress_bar.close()

    df = pd.DataFrame(all_reviews)
    df.to_csv(output_path, index=False)
    return df


def extract_sample_reviews(output_path: Path, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Descarga una muestra aleatoria balanceada de reseñas desde /reviews/sample.
    Ideal para iteraciones rápidas, pruebas de concepto y depuración inicial.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando muestra de {n:,} reseñas desde /reviews/sample (seed={seed})...")

    params = {"n": n, "seed": seed}
    res = requests.get(SAMPLE_ENDPOINT, params=params, timeout=60)
    res.raise_for_status()

    sample_data = res.json()
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    return df


def print_summary(df: pd.DataFrame, output_path: Path):
    """Muestra un reporte descriptivo del dataset descargado."""
    print("\n" + "=" * 60)
    print(">> REPORTE DE EXTRACCIÓN COMPLETADO")
    print("=" * 60)
    print(f"Archivo guardado en:      {output_path.resolve()}")
    print(f"Total de registros:       {len(df):,}")
    print(f"Columnas disponibles:     {list(df.columns)}")
    print(f"Memoria aproximada:       {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    if "label" in df.columns:
        print("\nDistribución de etiquetas (label 0 a 4):")
        dist = df["label"].value_counts().sort_index()
        for label, count in dist.items():
            pct = (count / len(df)) * 100
            print(f"  Calificación {label}: {count:6,} ({pct:5.1f}%)")

    print("\nPrimeras 2 reseñas de muestra:")
    for idx, row in df.head(2).iterrows():
        print(f"  [{row.get('id', idx)}] (Estrellas: {row.get('label')}): {str(row.get('text', ''))[:120]}...")
    print("=" * 60 + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extractor del dataset de reseñas de Amazon para el Desafío Taller II."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Ruta de destino del CSV (por defecto: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Límite máximo de registros a descargar (ej: 5000 para pruebas, omitir para descargar todo)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Tamaño de lote por solicitud HTTP (por defecto: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--sample-mode",
        action="store_true",
        help="Usar el endpoint /reviews/sample para descarga rápida de muestra representativa"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Cantidad de registros al usar --sample-mode (por defecto: 5000, máx: 5000)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignorar archivos parciales previos y comenzar la descarga desde cero"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_path = Path(args.output)

    if not check_server_status():
        sys.exit(1)

    if args.sample_mode:
        df = extract_sample_reviews(
            output_path=output_path,
            n=args.sample_size
        )
    else:
        df = extract_reviews_paginated(
            output_path=output_path,
            limit_total=args.limit,
            batch_size=args.batch_size,
            resume=not args.no_resume
        )

    print_summary(df, output_path)


if __name__ == "__main__":
    main()
