# Resumen de Representaciones y Vectorización de Texto

**Proyecto:** Desafío Taller II - Modelo de Análisis de Sentimiento (Strata)  
**Autor:** Alvaro Ybañez  
**Partición:** 134,187 reseñas en Train (80%) y 33,547 en Test (20%) estratificadas (seed=42).

---

## 📊 Comparativa de Tamaños de Vocabulario y Dimensiones

| Enfoque de Representación | Tamaño de Vocabulario | Dimensiones | Shape Train | Shape Test | Notas / Densidad |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **bow** | 11,149 | 11,149 | `(134187, 11149)` | `(33547, 11149)` | min_df=5, densidad=0.1328% |
| **tfidf_unigram** | 11,149 | 11,149 | `(134187, 11149)` | `(33547, 11149)` | min_df=5, densidad=0.1328% |
| **tfidf_bigram** | 72,978 | 72,978 | `(134187, 72978)` | `(33547, 72978)` | min_df=5, densidad=0.0313% |
| **embeddings_glove_100d** | 400,000 | 100 | `(134187, 100)` | `(33547, 100)` | cobertura de vocabulario: 87.5%, embeddings 100d promediados |
| **hf_pipeline (DistilBERT SST-2)** | 30,522 | 768 | `(134,187, 512)` | `-` | WordPiece subwords; media=42.6 tokens, truncadas(>512)=0.02% |

---

## 📌 Conclusiones y Decisiones de Modelado
1. **Partición sin fuga de información:** Todos los vectorizadores se ajustaron (`fit`) exclusivamente con el conjunto de Train.
2. **Eficiencia en memoria:** Las matrices BoW y TF-IDF se almacenaron en formato disperso comprimido `.npz` (`scipy.sparse`), reduciendo el uso de disco de varios gigabytes a apenas megabytes.
3. **Léxico y Embeddings:** GloVe 100d ofrece cobertura semántica global previa al entrenamiento, mientras que BoW y TF-IDF capturan términos específicos del dominio del corpus con un umbral `min_df=5`.
4. **Pipeline Hugging Face:** El tokenizador WordPiece maneja palabras fuera de vocabulario dividiendo en subpalabras y define una longitud máxima de 512 tokens para la etapa de inferencia con modelos basados en Transformers.
