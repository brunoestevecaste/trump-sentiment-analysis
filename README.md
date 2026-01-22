# 🇺🇸 Trump Tweets Sentiment Analysis with LSTM

Este proyecto realiza un análisis de sentimientos (Positivo, Negativo, Neutro) sobre una colección de tweets de Donald Trump. Utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) y redes neuronales profundas (**LSTM** y **Bi-LSTM**) implementadas en TensorFlow/Keras.

## 📌 Descripción del Proyecto

El objetivo es clasificar el tono emocional de los tweets basándose en un enfoque supervisado. 
El flujo de trabajo incluye:
1. **Etiquetado de Datos**: Generación de etiquetas (ground truth) usando un diccionario de palabras ponderadas (`trumpwords.xlsx`).
2. **Preprocesamiento**: Limpieza de texto, tokenización y padding.
3. **Modelado**: Implementación de redes neuronales recurrentes (RNN) utilizando arquitecturas LSTM y Bidireccional LSTM con Embeddings.
4. **Evaluación**: Análisis de métricas de precisión y matrices de confusión.

