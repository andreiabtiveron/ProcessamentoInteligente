import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_rgb_channels(image):
    # Separar canais
    r, g, b = cv2.split(image)

    # Equalizar cada canal
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    # Juntar os canais
    equalized = cv2.merge((r_eq, g_eq, b_eq))
    return equalized

# Caminho da imagem enviada
image_path = 'correcaoB.jpg'

# Ler imagem e converter para RGB
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Aplicar equalização por canal RGB
equalized_rgb = equalize_rgb_channels(img_rgb)

# Mostrar resultado
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_rgb)
plt.title('RGB Equalizado (R, G, B)')
plt.axis('off')

plt.tight_layout()
plt.show()
