import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "aula007.png"
image = cv2.imread(image_path)

# Converter a imagem para o espaço de cor HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir os limites para a cor verde no espaço HSV
lower_green = np.array([35, 50, 50])  # Limite inferior da cor verde
upper_green = np.array([85, 255, 255])  # Limite superior da cor verde

# Criar uma máscara para a cor verde
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Aplicar a máscara para extrair a região verde da imagem
result_green = cv2.bitwise_and(image, image, mask=mask_green)

# Plotar as imagens
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_green, cmap="gray")
plt.title("Máscara de Verde")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result_green, cv2.COLOR_BGR2RGB))
plt.title("Detecção Completa do Verde")
plt.axis("off")

plt.show()
