import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Variáveis globais
rect_start = None
rect_end = None
drawing = False
samples = []
labels = []
image = None
current_label = None
collected = 0
target_per_class = 0

# Função para desenhar o retângulo na imagem
def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, drawing, samples, labels, collected, target_per_class

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_start = (x, y)
        rect_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        rect_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_end = (x, y)
        x1, y1 = rect_start
        x2, y2 = rect_end

        # Coletar a amostra
        if x1 != x2 and y1 != y2:
            sample = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            if sample.size != 0:
                samples.append(sample)
                labels.append(current_label)
                collected += 1
                print(f"Amostra coletada para classe '{current_label}' ({collected}/{target_per_class})")

# Função para normalizar amostras
def normalize_samples(samples):
    norm_samples = []
    for sample in samples:
        sample_resized = cv2.resize(sample, (64, 64))
        norm_samples.append(sample_resized.flatten() / 255.0)
    return np.array(norm_samples)

def main():
    global image, current_label, collected, target_per_class

    image_path = os.path.join('/home', 'massarrahelenna', 'ProcessamentoInteligente', 'ProcessamentoInteligente', 'Aula-Pratica', 'Imagens', '4.jpg')
    
    image = cv2.imread(image_path)
    if image is None:
        print("Erro ao carregar a imagem. Verifique o caminho.")
        return

    num_classes = int(input("Digite o número de classes: "))
    target_per_class = int(input("Digite o número de amostras por classe: "))

    classes = [input(f"Digite o nome da classe {i+1}: ") for i in range(num_classes)]

    print(f"\nVocê vai coletar amostras para {num_classes} classes.")
    print(f"Cada classe terá {target_per_class} amostras.")

    # Redimensiona imagem se muito grande (evita travamento)
    max_width = 800
    max_height = 600
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    cv2.namedWindow("Imagem")
    cv2.setMouseCallback("Imagem", draw_rectangle)

    for current_label in classes:
        collected = 0
        print(f"\nColetando amostras para a classe '{current_label}'.")
        print("Desenhe retângulos com o mouse na janela da imagem.")

        while collected < target_per_class:
            img_copy = image.copy()
            if rect_start and rect_end:
                cv2.rectangle(img_copy, rect_start, rect_end, (0, 255, 0), 2)

            cv2.imshow("Imagem", img_copy)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

    if len(samples) == 0:
        print("Nenhuma amostra coletada.")
        return

    # Normalizar e treinar
    X = normalize_samples(samples)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"\nTotal de amostras coletadas: {len(samples)}")
    print(f"Distribuição das classes coletadas:")
    for label in classes:
        print(f"{label}: {labels.count(label)} amostras")

    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if len(X_train) < 1:
        print("Número insuficiente de amostras para treinar.")
        return

    # Ajustar n_neighbors
    n_neighbors = min(3, len(X_train))
    if n_neighbors < 1:
        print("Não é possível treinar o KNN com menos de 1 vizinho.")
        return

    # Treinamento
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)

    print(f"\nAcurácia do modelo KNN: {acc:.2f}")

#  # Salvar amostras e rótulos para uso posterior
#     np.savez("dados_coletados.txt", samples=np.array(samples, dtype=object), labels=np.array(labels))
#     print("Dados salvos em 'dados_coletados.txt'.")
if __name__ == "__main__":
    main()
    
