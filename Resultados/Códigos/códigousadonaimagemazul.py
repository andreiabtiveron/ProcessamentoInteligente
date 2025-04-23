import cv2
import numpy as np
import matplotlib.pyplot as plt

def open_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def split_rgb_channels(img):
    r, g, b = cv2.split(img)
    return r, g, b

def display_image(title, image):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1️⃣ Redução do canal azul com fator
def correct_blue_with_factor(r, g, b, factor=0.7):
    b_corrected = np.clip(b * factor, 0, 255).astype(np.uint8)
    return cv2.merge([r, g, b_corrected])

# 2️⃣ Equalização do canal azul
def equalize_blue_channel(r, g, b):
    b_eq = cv2.equalizeHist(b)
    return cv2.merge([r, g, b_eq])

# 3️⃣ Substituir canal azul pela média de R e G
def replace_blue_with_avg(r, g, b):
    b_avg = ((r.astype(np.float32) + g.astype(np.float32)) / 2).astype(np.uint8)
    return cv2.merge([r, g, b_avg])

# Função principal para aplicar o método escolhido
def apply_correction(image_path, method='factor'):
    img = open_image(image_path)
    r, g, b = split_rgb_channels(img)

    if method == 'factor':
        corrected_img = correct_blue_with_factor(r, g, b, factor=0.7)
        title = "Redução do Azul (fator 0.7)"
    elif method == 'equalize':
        corrected_img = equalize_blue_channel(r, g, b)
        title = "Equalização do Canal Azul"
    elif method == 'average':
        corrected_img = replace_blue_with_avg(r, g, b)
        title = "Substituição do Azul por Média R+G"
    else:
        print("Método inválido! Use 'factor', 'equalize' ou 'average'.")
        return

    # Exibe resultado
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Imagem Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(corrected_img)
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 🧪 Teste: Altere o caminho e o método desejado
image_path = 'Imagem.jpeg'  # substitua com o caminho correto
apply_correction(image_path, method='factor')     # ou 'equalize' ou 'average'
