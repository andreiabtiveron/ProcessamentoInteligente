import cv2
import numpy as np

img = cv2.imread("imagem_ajustada.jpg").astype(np.float32)

# Localiza o ponto mais brilhante da imagem (assumido como branco)
max_b, max_g, max_r = np.max(img[:,:,0]), np.max(img[:,:,1]), np.max(img[:,:,2])

# Normaliza os canais
img[:,:,0] *= 255.0 / max_b
img[:,:,1] *= 255.0 / max_g
img[:,:,2] *= 255.0 / max_r

# Clipa para intervalo [0,255] e converte de volta para uint8
img = np.clip(img, 0, 255).astype(np.uint8)

cv2.imwrite("imagem_white_patch.jpg", img)
