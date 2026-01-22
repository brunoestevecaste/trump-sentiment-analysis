from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional

def build_lstm_model(vocab_size, embedding_dim, input_length, num_classes):
    """Construye un modelo LSTM estándar."""
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_bilstm_model(vocab_size, embedding_dim, input_length, num_classes):
    """Construye un modelo Bidireccional LSTM (Mejorado)."""
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(SpatialDropout1D(0.4))
    # Bidirectional permite aprender contexto pasado y futuro
    model.add(Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model