import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


import seaborn as sns
import pandas as pd


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista de kernels a serem testados
kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    # Inicializar e treinar o classificador SVM com o kernel atual
    clf = svm.SVC(kernel=kernel, C=1.5)  
    clf.fit(X_train.reshape(len(X_train), -1), y_train)

    # Realizar previsões no conjunto de teste
    y_pred = clf.predict(X_test.reshape(len(X_test), -1))

    # Avaliar o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo SVM (Kernel {kernel}): {accuracy:.2f}')

    # Relatório de classificação
    print(f'Relatório de classificação (Kernel {kernel}):\n')
    print(classification_report(y_test, y_pred, zero_division=1))  # Definindo zero_division para 1
    print('-' * 40)

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
plt.show()