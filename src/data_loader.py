import pandas as pd
import numpy as np
import os

def load_and_label_data(tweets_path, words_path):
    """
    Carga los datos y genera las etiquetas basadas en el vocabulario.
    """
    if not os.path.exists(tweets_path) or not os.path.exists(words_path):
        raise FileNotFoundError("Por favor verifica que los archivos Excel estén en la carpeta data/")

    print(">>> Cargando datos...")
    tweets_df = pd.read_excel(tweets_path)
    words_df = pd.read_excel(words_path)

    # Agrupamos palabras por ID y sumamos sentimientos
    sentiment_counts = words_df.groupby('id')[['pos', 'neg']].sum().reset_index()

    # Lógica de etiquetado
    conditions = [
        sentiment_counts['pos'] > sentiment_counts['neg'],
        sentiment_counts['pos'] < sentiment_counts['neg']
    ]
    choices = ['positivo', 'negativo']
    sentiment_counts['etiqueta'] = np.select(conditions, choices, default='neutro')

    # Ajuste de índices para el merge (id archivo palabras = index tweet + 2)
    tweets_df['id'] = tweets_df.index + 2

    # Merge
    tweets_etiquetados = pd.merge(tweets_df, sentiment_counts, on='id', how='left')

    # Rellenar nulos
    tweets_etiquetados['pos'] = tweets_etiquetados['pos'].fillna(0)
    tweets_etiquetados['neg'] = tweets_etiquetados['neg'].fillna(0)
    tweets_etiquetados['etiqueta'] = tweets_etiquetados['etiqueta'].fillna('neutro')

    print(f">>> Datos cargados y etiquetados. Total filas: {len(tweets_etiquetados)}")
    return tweets_etiquetados