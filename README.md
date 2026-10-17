# Desafío Taller II - Modelo de Análisis de Sentimiento

**Materia:** Taller II — Ingeniería en Inteligencia Artificial (UNSTA)  
**Proyecto:** Desafío Taller II - Modelo de Análisis de Sentimiento (Strata)  
**Autor:** Alvaro Ybañez  

---

## 📌 Descripción del Proyecto

El objetivo del proyecto es construir un pipeline integral de Machine Learning para el análisis de sentimiento a partir de opiniones de clientes:
1. **Extracción y curaduría de datos:** Ingesta masiva desde la API REST de reseñas de Amazon.
2. **Preprocesamiento NLP:** Limpieza por expresiones regulares, normalización de contracciones, filtrado selectivo de stopwords (preservando negaciones), tokenización y lematización morfológica con etiquetas POS.
3. **Representación de texto (próximos pasos):** BoW, TF-IDF (unigramas y bigramas), embeddings pre-entrenados y pipelines de Hugging Face.
4. **Entrenamiento y Evaluación:** Modelos de clasificación evaluados sobre un banco de oraciones adversariales colaborativo curado por el curso.
5. **Puesta en producción:** Aplicación web para inferencia del modelo seleccionado.

---

## 📁 Estructura del Repositorio

```text
taller-ii-boilerplate/
├── data/
│   ├── raw/                 # Dataset crudo descargado de la API (ignorado en Git)
│   ├── processed/           # Dataset limpio preprocesado para modelado (ignorado en Git)
│   ├── vectorized/          # Matrices dispersas .npz y vectores .npy (ignorado en Git)
│   └── external/            # Vectores pre-entrenados GloVe (ignorado en Git)
├── models/                  # Vectorizadores entrenados .joblib (ignorado en Git)
├── notebooks/               # Análisis exploratorio (EDA) y experimentación
├── reports/                 # Tablas comparativas y reportes de entrega (Markdown)
├── src/
│   ├── __init__.py
│   ├── extract_reviews.py   # Script de ingesta masiva con control de reintentos y paginación
│   ├── preprocess.py        # Pipeline de limpieza NLP, lematización POS y binarización
│   └── vectorize.py         # Pipeline de vectorización (BoW, TF-IDF, GloVe, Hugging Face)
├── .gitignore               # Reglas de exclusión para datasets grandes, .venv y cachés
├── pyproject.toml           # Configuración del proyecto y dependencias de Poetry
├── poetry.lock              # Lockfile determinista de Poetry
└── README.md
```

---

## ⚙️ Configuración del Entorno con Poetry

El proyecto utiliza **Poetry** como gestor de dependencias y entornos virtuales.

### 1. Instalar dependencias

En la raíz del proyecto, ejecutá:

```bash
poetry install
```

Esto creará automáticamente el entorno virtual con todas las librerías necesarias (`requests`, `pandas`, `tqdm`, `nltk`, `scikit-learn`, `beautifulsoup4`).

---

## 📥 Paso 1: Extracción de Datos (`src/extract_reviews.py`)

La extracción se realiza contra la API desplegada en Render:  
`https://amazon-reviews-api-g5ae.onrender.com/reviews`

### Características de la extracción:
- **Manejo de Cold Start:** Consulta `/health` para esperar automáticamente a que la instancia de Render despierte sin fallar por timeout.
- **Paginación masiva resiliente:** Lotes de hasta 1.000 reseñas con barra de progreso interactiva (`tqdm`).
- **Tolerancia a fallos:** Reintentos automáticos con espera ante microcortes o errores 5xx del servidor.
- **Reanudación automática:** Si la descarga se interrumpe, al volver a correr el script detecta las filas ya guardadas y continúa desde el último `offset`.

### Comandos de ejecución:

**Descarga completa del dataset (210.000 reseñas):**
```bash
poetry run python src/extract_reviews.py
```

**Descarga de prueba / muestra acotada (ej. 5.000 reseñas):**
```bash
poetry run python src/extract_reviews.py --limit 5000
```

**Descarga rápida balanceada vía `/reviews/sample`:**
```bash
poetry run python src/extract_reviews.py --sample-mode --sample-size 5000
```

*El resultado se almacena en `data/raw/amazon_reviews_raw.csv`.*

---

## 🧹 Paso 2: Preprocesamiento NLP (`src/preprocess.py`)

El preprocesamiento implementa rigurosamente los principios teóricos vistos en **Clase 10 (Preprocesamiento I y II)**:

### Decisiones de diseño y justificación teórica:
1. **Normalización (Lowercasing):** Unificación a minúsculas para reducir la dispersión léxica.
2. **Expansión de contracciones:** Conversión de contracciones en inglés (`can't` $\rightarrow$ `can not`, `didn't` $\rightarrow$ `did not`, `won't` $\rightarrow$ `will not`) para evitar que la negación se pierda al eliminar puntuación.
3. **Limpieza secuencial con Regex:**
   - Remoción de etiquetas HTML (`<.*?>`).
   - Remoción de URLs y menciones (`@user`).
   - Normalización de caracteres repetidos (`looooove` $\rightarrow$ `love`).
   - Eliminación de dígitos y caracteres especiales de puntuación.
4. **Tratamiento crítico de Stopwords:**
   - Como se enfatiza en la teoría: **las negaciones (`not`, `no`, `never`, `nor`) alteran completamente la polaridad del texto**, por lo que **se excluyen explícitamente de la lista de stopwords a descartar**.
5. **Lematización morfológica con POS Tags:**
   - Se utiliza `WordNetLemmatizer` junto con `nltk.pos_tag` mapeando categorías gramaticales (verbos, sustantivos, adjetivos y adverbios). Esto permite transformar verbos en infinitivo (ej: `went` $\rightarrow$ `go`, `running` $\rightarrow$ `run`) y adjetivos a su forma base (ej: `better` $\rightarrow$ `good`).
6. **Binarización de polaridad:**
   - Calificaciones 0 y 1 $\rightarrow$ Sentimiento Negativo (`0`).
   - Calificación 2 $\rightarrow$ Sentimiento Neutro (descartado por defecto para maximizar la separabilidad en clasificación binaria).
   - Calificaciones 3 y 4 $\rightarrow$ Sentimiento Positivo (`1`).
7. **Reporte de Impacto:** Cálculo de métricas antes y después (reducción del vocabulario, Type-Token Ratio TTR, balance de clases y palabras más frecuentes por polaridad).

### Comandos de ejecución:

**Ejecución estándar completa:**
```bash
poetry run python src/preprocess.py
```

**Opciones adicionales:**
- `--no-lemmatization`: Saltea el paso de POS-tagging/lematización para ejecuciones ultrarrápidas de prueba.
- `--keep-neutrals`: Conserva las reseñas de calificación 2.
- `--min-tokens N`: Descarta reseñas que queden con menos de N palabras útiles tras el filtrado (por defecto: 2).

*El resultado se almacena en `data/processed/amazon_reviews_clean.csv`.*

---

## 🔢 Paso 3: Vectorización del Dataset (`src/vectorize.py`)

El script de vectorización implementa las 4 familias de representación numérica requeridas para la entrega:

### Características técnicas:
1. **Partición estratificada Train/Test (80/20):**
   - 134.187 reseñas para entrenamiento y 33.547 para evaluación.
   - Ajuste (`fit`) exclusivamente sobre el conjunto de Train para prevenir *data leakage*.
2. **Representaciones Dispersas (Scikit-Learn):**
   - **Bag of Words (BoW):** Unigramas con `min_df=5`. Vocabulario resultante: **11.149** términos.
   - **TF-IDF Unigrama:** Unigramas con `min_df=5`. Vocabulario resultante: **11.149** términos.
   - **TF-IDF Bigrama:** Unigramas y bigramas `(1, 2)` con `min_df=5`. Vocabulario resultante: **72.978** términos.
   - Persistencia de matrices comprimidas en formato `.npz` (`scipy.sparse`) en `data/vectorized/` y vectorizadores en `models/` (`.joblib`).
3. **Word Embeddings (GloVe 100d):**
   - Descarga e indexación de los vectores de Stanford (400.000 palabras en 100 dimensiones).
   - Representación densa de documentos calculando el promedio vectorial de las palabras conocidas (cobertura del **87.5%** sobre el vocabulario de Train).
   - Matrices guardadas en `data/vectorized/X_train_glove.npy` y `data/vectorized/X_test_glove.npy`.
4. **Pipeline Hugging Face (Transformers):**
   - Análisis del tokenizador contextual WordPiece (`distilbert-base-uncased-finetuned-sst-2-english`).
   - Vocabulario de subpalabras: **30.522** tokens.
   - Promedio de tokens por reseña: **42.6** (con truncado máximo a 512 tokens).

### Comandos de ejecución:

**Ejecución estándar completa:**
```bash
poetry run python src/vectorize.py
```

**Ejecución con muestra acotada para pruebas rápidas (ej. 5.000 reseñas):**
```bash
poetry run python src/vectorize.py --sample 5000
```

*Los resultados y métricas se resumen en el reporte [`reports/vectorization_summary.md`](reports/vectorization_summary.md).*

---

## 🗓️ Cronograma y Estado de Hitos

- [x] **28/09/2026 - Hito 1:** Extracción y preprocesamiento de reseñas para análisis de sentimiento.
- [x] **05/10/2026 - Hito 2:** Word Embeddings y conceptos de analogías semánticas.
- [x] **19/10/2026 - Hito 3:** Vectorización del dataset (BoW, TF-IDF unigrama/bigrama, embeddings, pipeline Hugging Face).
- [ ] **16/11/2026 - Hito 4:** Comparativa y evaluación de modelos sobre el test set adversarial.
- [ ] **Entrega Final:** Informe integrador y defensa técnica con cliente (Strata).