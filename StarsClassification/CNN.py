import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd

# Diretório contendo as imagens
data_dir = "data/"

# Inicialize listas para armazenar as imagens e labels
images = []
labels = []

# Função para carregar as imagens e extrair as labels
def load_images_and_labels(data_dir):
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = image / 255.0
                image = cv2.resize(image, (128, 128))
                images.append(image)
                labels.append(class_dir)

load_images_and_labels(data_dir)

# Converter listas em matrizes numpy
X = np.array(images)
y = np.array(labels)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Converter as strings em números inteiros
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Definir a arquitetura da CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Camada de Dropout
    layers.Dense(3, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=25, validation_split=0.2)

# Avaliar o desempenho do modelo no conjunto de teste
X_test = X_test / 255.0
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Relatório de classificação
report = classification_report(y_test, y_pred, target_names=['composite', 'plerion', 'shell'], zero_division=1)
print(report)

# Extrair métricas de treinamento
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Criar gráficos de perda e acurácia
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Obter a data e hora atuais
data_hora_atual = datetime.datetime.now()
data_hora_formatada = data_hora_atual.strftime('%Y_%m_%d_%Hh%Mm')
nome_do_arquivo = f'logs/report_{data_hora_formatada}.png'
plt.savefig(nome_do_arquivo)

# Calcular a matriz de confusão
confusion = confusion_matrix(y_test, y_pred)

# Definir as classes (rótulos)
classes = np.unique(y_test)

# Criar um DataFrame para visualização mais fácil
confusion_df = pd.DataFrame(confusion, index=classes, columns=classes)

# Visualizar a matriz de confusão usando seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_df, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Exibindo o gráfico
plt.show()
