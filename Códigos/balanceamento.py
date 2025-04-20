import cv2
import numpy as np

img = cv2.imread("correcaoB.jpg")
b, g, r = cv2.split(img)

# Balanceamento simples: iguala as médias dos canais
avg_b = np.mean(b)
avg_g = np.mean(g)
avg_r = np.mean(r)
avg = (avg_b + avg_g + avg_r) / 3

b = np.clip(b * (avg / avg_b), 0, 255).astype(np.uint8)
g = np.clip(g * (avg / avg_g), 0, 255).astype(np.uint8)
r = np.clip(r * (avg / avg_r), 0, 255).astype(np.uint8)

balanced = cv2.merge([b, g, r])
cv2.imwrite("imagem_corrigida.jpg", balanced)
