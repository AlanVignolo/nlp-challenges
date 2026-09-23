# Procesamiento de Lenguaje Natural I — CEIA, FIUBA

Cuatro desafíos de la materia Procesamiento de Lenguaje Natural I, de la Especialización en Inteligencia Artificial (FIUBA), cursada 2026. Trabajo individual.

## Desafío 1 — TF-IDF y Naïve Bayes

Vectorización de 20 Newsgroups con TF-IDF, similaridad coseno entre documentos y entre palabras (transponiendo la matriz documento-término), y clasificación con Naïve Bayes. Un clasificador por prototipos (1-NN sobre similaridad) da F1-macro 0.5050. Barriendo vectorizador, hiperparámetros y modelo, el mejor resultado es TF-IDF + ComplementNB con F1-macro 0.6999, contra 0.5854 del baseline.

## Desafío 2 — Embeddings propios con Word2Vec

Word2Vec (Gensim, skip-gram) entrenado sobre un corpus propio: tres obras del siglo XIX argentino (*Facundo*, *Una excursión a los indios ranqueles*, *Martín Fierro*), en vez del corpus de letras de canciones de la consigna original. Se inspeccionan vecinos semánticos de términos de interés y se proyectan los embeddings a 2D con t-SNE para identificar clusters (político, rural, numerales, plurales gramaticales).

## Desafío 3 — Modelo de lenguaje a nivel de caracteres

Modelo de lenguaje por caracteres sobre el mismo corpus del desafío 2, evaluado con perplejidad en validación. Compara SimpleRNN (200 unidades) contra LSTM (100 unidades) con cantidad de parámetros similar: perplejidad mínima 7.04 y 7.17 respectivamente. Incluye generación de texto con greedy search y beam search (determinista y estocástico, con distintas temperaturas).

## Desafío 4 — Traductor inglés→español seq2seq

Encoder-decoder LSTM con atención nula (seq2seq clásico), encoder inicializado con embeddings GloVe preentrenados. Se entrenan y comparan tres configuraciones de 64, 128 y 256 unidades. La de 256 da el mejor val_loss (0.5342) aunque la mejora sobre 128 (0.6044) es marginal frente al costo de entrenamiento.

Decisión deliberada sobre el largo de las secuencias: el percentil 95 de longitud rondaba los 11-12 tokens, pero se optó por truncar en 35 (input) y 40 (output) en vez de en el percentil 95, para no descartar las oraciones largas — son las que exigen generalización real al modelo, aunque impliquen mucho padding en las oraciones cortas.

## Cómo correrlos

Los datasets, los pesos entrenados (`.keras`) y los outputs (embeddings, historiales de perplejidad) no están versionados en este repo — quedan afuera por peso y porque se regeneran corriendo el notebook. Cada notebook descarga o genera lo que necesita en sus primeras celdas: 20 Newsgroups vía scikit-learn, el corpus de literatura argentina (`corpus_ar/`) hay que agregarlo a mano si no está, y el par de oraciones inglés-español se descarga solo.

El desafío 3 espera encontrar `my_model.keras` (y el desafío 4 entrena los suyos desde cero) para no reentrenar; si no existen, hay que poner `RETRAIN` / `RETRAIN_LSTM` en `True` para generarlos. El desafío 4 depende además de un archivo de embeddings GloVe (`gloveembedding.pkl`) que descarga de Google Drive en tiempo de ejecución.
