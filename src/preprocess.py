import argparse
import re
import string
import sys
from collections import Counter
from pathlib import Path
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm

# Asegurar recursos de NLTK instalados
for resource in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng"
]:
    nltk.download(resource, quiet=True)

DEFAULT_RAW_PATH = Path("data/raw/amazon_reviews_raw.csv")
DEFAULT_PROCESSED_PATH = Path("data/processed/amazon_reviews_clean.csv")

CONTRACTIONS = {
    r"\bcan'?t\b": "can not",
    r"\bwon'?t\b": "will not",
    r"\bdon'?t\b": "do not",
    r"\bdidn'?t\b": "did not",
    r"\bdoesn'?t\b": "does not",
    r"\bisn'?t\b": "is not",
    r"\baren'?t\b": "are not",
    r"\bwasn'?t\b": "was not",
    r"\bweren'?t\b": "were not",
    r"\bhasn'?t\b": "has not",
    r"\bhaven'?t\b": "have not",
    r"\bhadn'?t\b": "had not",
    r"\bwouldn'?t\b": "would not",
    r"\bshouldn'?t\b": "should not",
    r"\bcouldn'?t\b": "could not",
    r"\bain'?t\b": "is not",
    r"\bit'?s\b": "it is",
    r"\bi'?m\b": "i am",
    r"\bthey'?re\b": "they are",
    r"\byou'?re\b": "you are"
}

EXTENDED_PUNCTUATION = string.punctuation + "“”‘’«»—–…¿¡！？：，（）"
PUNCT_TABLE = str.maketrans(EXTENDED_PUNCTUATION, " " * len(EXTENDED_PUNCTUATION))

NEGATION_WORDS = {"not", "no", "never", "nor", "neither", "hardly", "barely"}

BASE_STOPWORDS = set(stopwords.words("english")) - NEGATION_WORDS

LEMMATIZER = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Mapea etiquetas POS de Penn Treebank a categorías de WordNet.
    Visto en Clase 10: 'Lematización y POS tags'.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text_regex(text: str) -> str:
    """
    Limpia texto crudo aplicando normalizaciones secuenciales:
    1. Conversión a minúsculas.
    2. Expansión de contracciones para preservar negaciones.
    3. Remoción de etiquetas HTML (<.*?>).
    4. Remoción de URLs y menciones (@usuario).
    5. Normalización de caracteres repetidos ('looooove' -> 'love').
    6. Remoción de dígitos numéricos.
    7. Remoción de puntuación.
    8. Normalización de espacios en blanco múltiples.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    text = re.sub(r"\d+", " ", text)

    text = text.translate(PUNCT_TABLE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, apply_lemmatization: bool = True) -> str:
    """
    Tokeniza, filtra stopwords preservando negaciones y lematiza según POS tag.
    """
    tokens = word_tokenize(text)

    filtered = [w for w in tokens if (w not in BASE_STOPWORDS and len(w) > 1)]

    if not apply_lemmatization or not filtered:
        return " ".join(filtered)

    tagged_tokens = pos_tag(filtered)
    lemmas = [
        LEMMATIZER.lemmatize(token, pos=get_wordnet_pos(tag))
        for token, tag in tagged_tokens
    ]
    return " ".join(lemmas)


def preprocess_text(text: str, apply_lemmatization: bool = True) -> str:
    """Función de pipeline integral para procesar un documento individual."""
    cleaned = clean_text_regex(text)
    if not cleaned:
        return ""
    return tokenize_and_lemmatize(cleaned, apply_lemmatization=apply_lemmatization)


def calculate_vocabulary_metrics(series: pd.Series) -> dict:
    """Calcula estadísticas del vocabulario: total de palabras, vocabulario único y TTR."""
    total_tokens = 0
    vocab = set()
    for text in series:
        words = str(text).split()
        total_tokens += len(words)
        vocab.update(words)

    vocab_size = len(vocab)
    ttr = (vocab_size / total_tokens) if total_tokens > 0 else 0.0
    return {
        "total_tokens": total_tokens,
        "vocabulary_size": vocab_size,
        "ttr": ttr
    }


def compute_top_words(series: pd.Series, top_k: int = 15) -> list[tuple[str, int]]:
    """Encuentra las palabras más frecuentes en un corpus de texto."""
    counter = Counter()
    for text in series:
        counter.update(str(text).split())
    return counter.most_common(top_k)


def generate_impact_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    initial_len: int,
    output_path: Path
):
    """
    Genera el reporte de impacto antes y después del procesamiento
    (Requisito explícito de la teoría en Clase 10 - Slide 12).
    """
    print("\n" + "=" * 65)
    print(">> REPORTE DE IMPACTO: PREPROCESAMIENTO NLP")
    print("=" * 65)

    print(f"\n1. Depuración y Balance de Filas:")
    print(f"  • Registros originales:         {initial_len:,}")
    print(f"  • Registros tras binarización:  {len(df_raw):,}")
    print(f"  • Registros finales limpios:    {len(df_clean):,} (Descartes vacíos: {len(df_raw) - len(df_clean):,})")

    print(f"\n2. Distribución de clases de sentimiento:")
    dist = df_clean["sentiment"].value_counts().sort_index()
    for s_val, count in dist.items():
        name = "Positiva (1)" if s_val == 1 else "Negativa (0)"
        pct = (count / len(df_clean)) * 100
        print(f"  • Clase {name}: {count:6,} ({pct:5.1f}%)")

    print(f"\n3. Impacto en Vocabulario y Reducción Léxica:")
    raw_metrics = calculate_vocabulary_metrics(df_raw["text"])
    clean_metrics = calculate_vocabulary_metrics(df_clean["cleaned_text"])

    print(f"  • Total de tokens (Original):     {raw_metrics['total_tokens']:,}")
    print(f"  • Total de tokens (Procesado):    {clean_metrics['total_tokens']:,} (Reducción: {(1 - clean_metrics['total_tokens']/raw_metrics['total_tokens'])*100:.1f}%)")
    print(f"  • Vocabulario único (Original):   {raw_metrics['vocabulary_size']:,}")
    print(f"  • Vocabulario único (Procesado):  {clean_metrics['vocabulary_size']:,} (Reducción: {(1 - clean_metrics['vocabulary_size']/raw_metrics['vocabulary_size'])*100:.1f}%)")
    print(f"  • TTR (Type-Token Ratio) Final:   {clean_metrics['ttr']:.4f}")

    print(f"\n4. Top Palabras Más Representativas por Sentimiento:")
    pos_top = compute_top_words(df_clean[df_clean["sentiment"] == 1]["cleaned_text"], top_k=10)
    neg_top = compute_top_words(df_clean[df_clean["sentiment"] == 0]["cleaned_text"], top_k=10)

    print("  • Positivas (Top 10):", ", ".join(f"{w} ({c})" for w, c in pos_top))
    print("  • Negativas (Top 10):", ", ".join(f"{w} ({c})" for w, c in neg_top))

    print(f"\n Archivo final guardado exitosamente en: {output_path.resolve()}")
    print("=" * 65 + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pipeline de preprocesamiento y limpieza de texto NLP para reseñas de Amazon."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_RAW_PATH),
        help=f"Ruta del archivo CSV crudo (por defecto: {DEFAULT_RAW_PATH})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_PROCESSED_PATH),
        help=f"Ruta de destino del CSV procesado (por defecto: {DEFAULT_PROCESSED_PATH})"
    )
    parser.add_argument(
        "--keep-neutrals",
        action="store_true",
        help="Mantener reseñas neutras (calificación 2). Por defecto se descartan para análisis binario"
    )
    parser.add_argument(
        "--no-lemmatization",
        action="store_true",
        help="Omitir lematización y POS tagging (útil para acelerar si solo se desea tokenización)"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=2,
        help="Cantidad mínima de palabras que debe conservar una reseña tras la limpieza (por defecto: 2)"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f" Error: No se encontró el dataset crudo en '{input_path}'.")
        print(" Ejecutá primero 'poetry run python src/extract_reviews.py' para descargarlo.")
        sys.exit(1)

    print(f"Cargando dataset crudo desde: {input_path} ...")
    df = pd.read_csv(input_path)
    initial_len = len(df)
    print(f"Registros cargados: {initial_len:,}")

    if not args.keep_neutrals:
        print("Descartando reseñas neutras (label == 2) para clasificación binaria de polaridad...")
        df = df[df["label"] != 2].copy()

    df["sentiment"] = df["label"].apply(lambda x: 1 if x >= 3 else 0)

    apply_lemmatization = not args.no_lemmatization
    lemmatization_str = "Activada (POS Tagging + WordNet)" if apply_lemmatization else "Desactivada"
    print(f"Iniciando preprocesamiento de texto (Lematización: {lemmatization_str})...")

    tqdm.pandas(desc="Limpiando y normalizando texto", unit="docs")
    df["cleaned_text"] = df["text"].progress_apply(
        lambda t: preprocess_text(t, apply_lemmatization=apply_lemmatization)
    )
    df = df[df["cleaned_text"].str.split().str.len() >= args.min_tokens].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    generate_impact_report(
        df_raw=df[["text", "label", "sentiment"]],
        df_clean=df,
        initial_len=initial_len,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
