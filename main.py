import os
from sklearn.model_selection import train_test_split
from src.data_loader import load_and_label_data
from src.preprocessing import prepare_data_for_lstm
from src.models import build_lstm_model, build_bilstm_model
from src.visualization import plot_training_history, evaluate_model

# --- CONFIGURACIÓN ---
DATA_PATH = "data/2016_12_05-TrumpTwitterAll.xlsx"
WORDS_PATH = "data/trumpwords.xlsx"
MAX_WORDS = 5000
MAX_LEN = 50
EMBEDDING_DIM = 100
EPOCHS = 5
BATCH_SIZE = 64

def main():
    # 1. Cargar Datos
    print("--- 1. INICIANDO PROCESO DE CARGA ---")
    df = load_and_label_data(DATA_PATH, WORDS_PATH)
    
    # 2. Preprocesamiento
    print("\n--- 2. PREPARANDO DATOS PARA MODELO ---")
    X, Y, tokenizer, le = prepare_data_for_lstm(df, 'Tweet', 'etiqueta', MAX_WORDS, MAX_LEN)
    
    # Split Train/Test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # 3. Construcción y Entrenamiento del Modelo
    print("\n--- 3. ENTRENANDO MODELO (Bidireccional LSTM) ---")
    # Puedes cambiar a build_lstm_model si prefieres el modelo simple
    model = build_bilstm_model(vocab_size=MAX_WORDS, 
                               embedding_dim=EMBEDDING_DIM, 
                               input_length=MAX_LEN, 
                               num_classes=3) # 3 clases: Pos, Neg, Neu
    
    print(model.summary())
    
    history = model.fit(X_train, Y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=0.1,
                        verbose=1)
    
    # 4. Evaluación y Visualización
    print("\n--- 4. EVALUACIÓN DE RESULTADOS ---")
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Precisión Final en Test: {accuracy:.4f}")
    
    plot_training_history(history)
    evaluate_model(model, X_test, Y_test, le)

if __name__ == "__main__":
    main()