import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Diretório contendo as imagens
data_dir = "data/"

# Inicialize listas para armazenar as imagens e labels
images = []
labels = []

# Função para carregar as imagens e extrair as labels
def load_images_and_labels():
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                # Carregando a imagem em tons de cinza (você pode ajustar conforme necessário)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # Normalização dos valores de pixel para o intervalo [0, 1]
                image = image / 255.0  
                # Redimensionando a imagem para um tamanho fixo (por exemplo, 128x128 pixels)
                image = cv2.resize(image, (128, 128))
                images.append(image)
                labels.append(class_dir)

load_images_and_labels()
# Converter listas em matrizes numpy
X = np.array(images)
y = np.array(labels)

# Dividir os dados em conjuntos de treinamento e teste
X_train,  X_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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
    layers.Dense(3, activation='softmax')  #Define a ultima camada com a quantidade de neurônios correspondente ao numero de classes
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=25, validation_split=0.2)  # Usando validação cruzada

# Avaliar o desempenho do modelo no conjunto de teste
X_test = X_test / 255.0  # Normalização
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

# Criar um gráfico de perda
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Criar um gráfico de acurácia
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')



# Obter a data e hora atuais
data_hora_atual = datetime.datetime.now()
# Formatar a data e hora no formato desejado (por exemplo, YYYYMMDD_HHMMSS)
data_hora_formatada = data_hora_atual.strftime('%Y%m%d_%H%M%S')
# Nome do arquivo com data e hora
nome_do_arquivo = f'figura_{data_hora_formatada}.png'
# Salvar a figura em um arquivo com data e hora no nome
plt.savefig(nome_do_arquivo)
# Exibindo o gráfico
plt.show()