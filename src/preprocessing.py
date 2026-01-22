import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def clean_text(text):
    """Limpieza básica de texto para NLP."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def prepare_data_for_lstm(df, text_col, label_col, max_words=5000, max_len=50):
    """
    Tokeniza los textos y codifica las etiquetas.
    Retorna: X (padded), Y (one-hot), tokenizer, label_encoder
    """
    print(">>> Preprocesando textos y etiquetas...")
    
    # 1. Limpieza
    df['clean_text'] = df[text_col].apply(clean_text)
    
    # 2. Tokenización
    tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['clean_text'].values)
    
    X = tokenizer.texts_to_sequences(df['clean_text'].values)
    X = pad_sequences(X, maxlen=max_len)
    
    # 3. Codificación de etiquetas
    le = LabelEncoder()
    Y_integers = le.fit_transform(df[label_col])
    Y = to_categorical(Y_integers)
    
    print(f"Dimensiones de X: {X.shape}")
    print(f"Clases detectadas: {le.classes_}")
    
    return X, Y, tokenizer, le