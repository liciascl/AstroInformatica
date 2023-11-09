import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Convertendo as listas em matrizes numpy
images = np.array(images)
labels = np.array(labels)

# Dividindo o dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Inicializando e treinando o classificador Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train.reshape(len(X_train), -1), y_train)

# Realizando previsões no conjunto de teste
y_pred = clf.predict(X_test.reshape(len(X_test), -1))


# Número de imagens aleatórias para testar
num_images_to_test = 5
# Preparar a figura para plotagem
fig, axes = plt.subplots(num_images_to_test, 2, figsize=(10, 20))



for i in range(num_images_to_test):
    random_index = random.randint(0, len(X_test) - 1)
    random_test_image = X_test[random_index].reshape(1, -1)
    true_label = y_test[random_index]

    # Classificando a imagem com o modelo treinado
    predicted_class = clf.predict(random_test_image)

    # Realizar previsões no conjunto de teste
    predicted_class = clf.predict(X_test.reshape(len(X_test), -1))

    # Exibindo a imagem
    axes[i, 0].imshow(X_test[random_index], cmap='gray')
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f"Classe Verdadeira: {true_label}")

    # Exibindo a classe prevista
    axes[i, 1].text(0.5, 0.5, f"Classe Prevista: {predicted_class[0]}", fontsize=12, ha='center')
    axes[i, 1].axis('off')


# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo : {accuracy:.2f}')

# Relatório de classificação
print(f'Relatório de classificação):\n')
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

plt.tight_layout()
plt.show()
