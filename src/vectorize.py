import argparse
import sys
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_PROCESSED_PATH = Path("data/processed/amazon_reviews_clean.csv")
DEFAULT_SPLIT_PATH = Path("data/processed/train_test_split.csv")
DEFAULT_VECTORIZED_DIR = Path("data/vectorized")
DEFAULT_EXTERNAL_DIR = Path("data/external")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_REPORT_PATH = Path("reports/vectorization_summary.md")

TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_DF = 5

GLOVE_DIM = 100
GLOVE_FILENAME = f"glove.6B.{GLOVE_DIM}d.txt"
GLOVE_DIRECT_URL = "https://huggingface.co/arjahojnik/LSTM-sentiment-model/resolve/af8ba3a2939dd573a5d9384efecbe7f04137860b/glove.6B.100d.txt"
GLOVE_ZIP_URL = "https://nlp.stanford.edu/data/glove.6B.zip"

HF_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
HF_MAX_LENGTH = 512

SPARSE_VECTORIZERS = {
    "bow": lambda: CountVectorizer(min_df=MIN_DF),
    "tfidf_unigram": lambda: TfidfVectorizer(min_df=MIN_DF, ngram_range=(1, 1)),
    "tfidf_bigram": lambda: TfidfVectorizer(min_df=MIN_DF, ngram_range=(1, 2)),
}


def load_and_validate_dataset(csv_path: Path, sample_size: int | None = None) -> pd.DataFrame:
    """Carga y valida el dataset preprocesado."""
    if not csv_path.exists():
        print(f"Error: No se encontró el dataset en '{csv_path}'.")
        print("Ejecutá previamente 'poetry run python src/preprocess.py'.")
        sys.exit(1)

    print(f"Cargando dataset desde: {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Filtrar posibles valores nulos o cadenas vacías
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)

    if sample_size and sample_size < len(df):
        print(f"Modo muestra acotada activo: usando {sample_size:,} reseñas aleatorias balanceadas.")
        n_half = sample_size // 2
        pos_df = df[df["sentiment"] == 1].sample(n=n_half, random_state=RANDOM_STATE)
        neg_df = df[df["sentiment"] == 0].sample(n=n_half, random_state=RANDOM_STATE)
        df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Dataset listo: {len(df):,} reseñas.")
    return df


def split_dataset(
    df: pd.DataFrame,
    split_path: Path,
    vectorized_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el corpus en conjuntos estratificados de Train (80%) y Test (20%).
    Guarda los identificadores y etiquetas para garantizar reproducibilidad exacta.
    """
    print(f"Realizando partición Train/Test estratificada ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})...")
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["sentiment"],
        random_state=RANDOM_STATE
    )

    split_path.parent.mkdir(parents=True, exist_ok=True)
    vectorized_dir.mkdir(parents=True, exist_ok=True)

    # Guardar mapeo de splits
    split_records = pd.concat([
        pd.DataFrame({"id": train_df["id"], "split": "train"}),
        pd.DataFrame({"id": test_df["id"], "split": "test"})
    ])
    split_records.to_csv(split_path, index=False)

    # Guardar vectores objetivo y
    np.save(vectorized_dir / "y_train.npy", train_df["sentiment"].to_numpy())
    np.save(vectorized_dir / "y_test.npy", test_df["sentiment"].to_numpy())

    print(f"  • Train: {len(train_df):,} registros | Test: {len(test_df):,} registros.")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def vectorize_sparse_representation(
    name: str,
    train_texts: pd.Series,
    test_texts: pd.Series,
    vectorized_dir: Path,
    models_dir: Path
) -> dict:
    """
    Ajusta (fit) el vectorizador exclusivamente con Train y transforma Train y Test.
    Almacena las matrices dispersas en formato comprimido .npz y el modelo en .joblib.
    """
    print(f"\nVectorizando con {name.upper()}...")
    vectorizer = SPARSE_VECTORIZERS[name]()

    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    vectorized_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Guardado de matrices dispersas comprimidas (.npz)
    sparse.save_npz(vectorized_dir / f"X_train_{name}.npz", x_train)
    sparse.save_npz(vectorized_dir / f"X_test_{name}.npz", x_test)

    # Guardado del vectorizador ajustado para inferencia posterior
    joblib.dump(vectorizer, models_dir / f"vectorizer_{name}.joblib")

    vocab_size = len(vectorizer.vocabulary_)
    density = x_train.nnz / (x_train.shape[0] * x_train.shape[1])

    print(f"  Vocabulario: {vocab_size:,} términos | Dimensiones: {x_train.shape[1]:,}")
    print(f"  Shape Train: {x_train.shape} | Shape Test: {x_test.shape} | Densidad: {density:.4%}")

    return {
        "representation": name,
        "vocabulary_size": vocab_size,
        "dimensions": x_train.shape[1],
        "train_shape": str(x_train.shape),
        "test_shape": str(x_test.shape),
        "notes": f"min_df={MIN_DF}, densidad={density:.4%}"
    }


def download_glove_vectors(external_dir: Path) -> Path:
    """
    Descarga los vectores pre-entrenados GloVe (100d) con barra de progreso.
    Intenta primero la descarga directa del archivo y luego el ZIP oficial de Stanford.
    """
    external_dir.mkdir(parents=True, exist_ok=True)
    glove_path = external_dir / GLOVE_FILENAME

    if glove_path.exists() and glove_path.stat().st_size > 100 * 1024 * 1024:
        print(f"Vectores GloVe ya disponibles en: {glove_path}")
        return glove_path

    print(f"Descargando vectores pre-entrenados GloVe 100d (~340 MB)...")

    # Intento 1: Mirror directo de Hugging Face
    try:
        with requests.get(GLOVE_DIRECT_URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(glove_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Descargando GloVe"
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        print("Descarga directa de GloVe completada.")
        return glove_path
    except Exception as e:
        print(f"Aviso: Descarga directa falló ({e}). Intentando desde archivo ZIP oficial...")

    # Intento 2: ZIP oficial de Stanford
    zip_path = external_dir / "glove.6B.zip"
    with requests.get(GLOVE_ZIP_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Descargando glove.6B.zip"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    print(f"Extrayendo '{GLOVE_FILENAME}' del archivo ZIP...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extract(GLOVE_FILENAME, external_dir)
    zip_path.unlink()

    return glove_path


def load_glove_embeddings(glove_file: Path) -> tuple[dict[str, int], np.ndarray]:
    """Carga los vectores GloVe en memoria estructurada de NumPy."""
    words = []
    vectors = []
    print(f"Cargando vectores GloVe desde {glove_file.name}...")
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Indexando GloVe", unit=" palabras"):
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vectors.append(np.asarray(parts[1:], dtype=np.float32))

    word_to_index = {word: i for i, word in enumerate(words)}
    matrix = np.vstack(vectors)
    return word_to_index, matrix


def embed_corpus_documents(
    texts: list[str],
    word_to_index: dict[str, int],
    matrix: np.ndarray
) -> np.ndarray:
    """
    Representa cada reseña promediando los vectores GloVe de sus palabras conocidas.
    Enfoque visto en clase de Embeddings para oraciones/documentos completos.
    """
    dim = matrix.shape[1]
    result = np.zeros((len(texts), dim), dtype=np.float32)

    for i, text in enumerate(tqdm(texts, desc="Promediando vectores", unit="docs")):
        indices = [word_to_index[w] for w in text.split() if w in word_to_index]
        if indices:
            result[i] = matrix[indices].mean(axis=0)

    return result


def vectorize_glove_representation(
    train_texts: list[str],
    test_texts: list[str],
    external_dir: Path,
    vectorized_dir: Path
) -> dict:
    """Genera las representaciones densas con embeddings GloVe."""
    print("\nVectorizando con embeddings pre-entrenados GloVe (100d)...")
    glove_path = download_glove_vectors(external_dir)
    word_to_index, matrix = load_glove_embeddings(glove_path)

    x_train = embed_corpus_documents(train_texts, word_to_index, matrix)
    x_test = embed_corpus_documents(test_texts, word_to_index, matrix)

    np.save(vectorized_dir / "X_train_glove.npy", x_train)
    np.save(vectorized_dir / "X_test_glove.npy", x_test)

    # Cobertura léxica del corpus
    corpus_vocab = {w for text in train_texts for w in text.split()}
    known_words = sum(1 for w in corpus_vocab if w in word_to_index)
    coverage_pct = (known_words / len(corpus_vocab)) * 100 if corpus_vocab else 0.0

    print(f"  Vocabulario GloVe: {len(word_to_index):,} palabras | Dimensiones: {matrix.shape[1]}")
    print(f"  Cobertura léxica en Train: {known_words:,}/{len(corpus_vocab):,} ({coverage_pct:.1f}%)")

    return {
        "representation": "embeddings_glove_100d",
        "vocabulary_size": len(word_to_index),
        "dimensions": matrix.shape[1],
        "train_shape": str(x_train.shape),
        "test_shape": str(x_test.shape),
        "notes": f"cobertura de vocabulario: {coverage_pct:.1f}%, embeddings 100d promediados"
    }


def analyze_huggingface_tokenizer(train_texts: list[str]) -> dict:
    """
    Analiza el tokenizador contextual del pipeline de Hugging Face.
    Mide el tamaño de vocabulario de subpalabras (WordPiece/BPE) y estadísticas de longitud.
    """
    print(f"\nAnalizando tokenizador del pipeline de Hugging Face ({HF_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    lengths = []
    batch_size = 2000
    for i in tqdm(range(0, len(train_texts), batch_size), desc="Tokenizando muestra HF", unit="lotes"):
        batch = train_texts[i : i + batch_size]
        encodings = tokenizer(batch, truncation=False)
        lengths.extend([len(ids) for ids in encodings["input_ids"]])

    lengths_arr = np.array(lengths)
    truncated_pct = (lengths_arr > HF_MAX_LENGTH).mean() * 100

    print(f"  Vocabulario HF: {tokenizer.vocab_size:,} subpalabras | Dimensión base: 768")
    print(f"  Tokens promedio por reseña: {lengths_arr.mean():.1f} (máx: {lengths_arr.max()})")
    print(f"  Reseñas que exceden {HF_MAX_LENGTH} tokens: {truncated_pct:.2f}%")

    return {
        "representation": "hf_pipeline (DistilBERT SST-2)",
        "vocabulary_size": tokenizer.vocab_size,
        "dimensions": 768,
        "train_shape": f"({len(train_texts):,}, {HF_MAX_LENGTH})",
        "test_shape": "-",
        "notes": f"WordPiece subwords; media={lengths_arr.mean():.1f} tokens, truncadas(>{HF_MAX_LENGTH})={truncated_pct:.2f}%"
    }


def write_vectorization_report(
    rows: list[dict],
    n_train: int,
    n_test: int,
    report_path: Path
):
    """Genera el reporte en Markdown con la tabla comparativa de vocabularios."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Resumen de Representaciones y Vectorización de Texto",
        "",
        f"**Proyecto:** Desafío Taller II - Modelo de Análisis de Sentimiento (Strata)  ",
        f"**Autor:** Alvaro Ybañez  ",
        f"**Partición:** {n_train:,} reseñas en Train (80%) y {n_test:,} en Test (20%) estratificadas (seed={RANDOM_STATE}).",
        "",
        "---",
        "",
        "## 📊 Comparativa de Tamaños de Vocabulario y Dimensiones",
        "",
        "| Enfoque de Representación | Tamaño de Vocabulario | Dimensiones | Shape Train | Shape Test | Notas / Densidad |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for r in rows:
        lines.append(
            f"| **{r['representation']}** | {r['vocabulary_size']:,} | {r['dimensions']:,} | "
            f"`{r['train_shape']}` | `{r['test_shape']}` | {r['notes']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 📌 Conclusiones y Decisiones de Modelado",
        "1. **Partición sin fuga de información:** Todos los vectorizadores se ajustaron (`fit`) exclusivamente con el conjunto de Train.",
        "2. **Eficiencia en memoria:** Las matrices BoW y TF-IDF se almacenaron en formato disperso comprimido `.npz` (`scipy.sparse`), reduciendo el uso de disco de varios gigabytes a apenas megabytes.",
        "3. **Léxico y Embeddings:** GloVe 100d ofrece cobertura semántica global previa al entrenamiento, mientras que BoW y TF-IDF capturan términos específicos del dominio del corpus con un umbral `min_df=5`.",
        "4. **Pipeline Hugging Face:** El tokenizador WordPiece maneja palabras fuera de vocabulario dividiendo en subpalabras y define una longitud máxima de 512 tokens para la etapa de inferencia con modelos basados en Transformers.",
        ""
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReporte comparativo generado con éxito en: {report_path.resolve()}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pipeline de vectorización de texto (BoW, TF-IDF unigrama/bigrama, Embeddings y Hugging Face)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_PROCESSED_PATH),
        help=f"Ruta del dataset limpio CSV (por defecto: {DEFAULT_PROCESSED_PATH})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_VECTORIZED_DIR),
        help=f"Directorio donde guardar las matrices vectorizadas (por defecto: {DEFAULT_VECTORIZED_DIR})"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help=f"Directorio donde guardar los vectorizadores entrenados (por defecto: {DEFAULT_MODELS_DIR})"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help=f"Ruta del archivo de reporte Markdown (por defecto: {DEFAULT_REPORT_PATH})"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Cantidad de reseñas a muestrear para pruebas de velocidad (ej: 5000)"
    )
    parser.add_argument(
        "--skip-glove",
        action="store_true",
        help="Omitir la descarga y cálculo de embeddings GloVe"
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Omitir el análisis del tokenizador de Hugging Face"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    csv_path = Path(args.input)
    vectorized_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    report_path = Path(args.report)
    split_path = Path(DEFAULT_SPLIT_PATH)
    external_dir = Path(DEFAULT_EXTERNAL_DIR)

    # 1. Cargar y particionar dataset
    df = load_and_validate_dataset(csv_path, sample_size=args.sample)
    train_df, test_df = split_dataset(df, split_path, vectorized_dir)

    rows = []

    # 2. Representaciones dispersas: BoW, TF-IDF unigrama, TF-IDF bigrama
    for name in ["bow", "tfidf_unigram", "tfidf_bigram"]:
        res = vectorize_sparse_representation(
            name=name,
            train_texts=train_df["cleaned_text"],
            test_texts=test_df["cleaned_text"],
            vectorized_dir=vectorized_dir,
            models_dir=models_dir
        )
        rows.append(res)

    # 3. Representación con Embeddings GloVe (100d)
    if not args.skip_glove:
        glove_res = vectorize_glove_representation(
            train_texts=train_df["cleaned_text"].tolist(),
            test_texts=test_df["cleaned_text"].tolist(),
            external_dir=external_dir,
            vectorized_dir=vectorized_dir
        )
        rows.append(glove_res)

    # 4. Pipeline de Hugging Face (análisis de tokenizador y vocabulario)
    if not args.skip_hf:
        hf_res = analyze_huggingface_tokenizer(train_df["text"].tolist())
        rows.append(hf_res)

    # 5. Generar reporte comparativo
    write_vectorization_report(
        rows=rows,
        n_train=len(train_df),
        n_test=len(test_df),
        report_path=report_path
    )

    print("\n" + "=" * 60)
    print(">> PIPELINE DE VECTORIZACIÓN FINALIZADO CON ÉXITO")
    print("=" * 60)
    for r in rows:
        print(f" • {r['representation']:<30} Vocabulario: {r['vocabulary_size']:>8,} | Dims: {r['dimensions']:>7,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
