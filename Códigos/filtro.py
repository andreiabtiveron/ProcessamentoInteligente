import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 📥 Ler a imagem
img = cv.imread('Figura_ajustada.png')

# Verificar se a imagem foi lida corretamente
if img is None:
    print("Erro ao carregar a imagem! Verifique o caminho.")
    exit()

# ➕ Adicionar ruído (opcional, só para fins de teste)
noise = np.random.randn(*img.shape) * 10
noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

# 🧼 Aplicar o filtro de remoção de ruído
denoised = cv.fastNlMeansDenoisingColored(noisy, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

# 🖼️ Mostrar as imagens
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('Original'), plt.axis('off')
plt.subplot(132), plt.imshow(cv.cvtColor(noisy, cv.COLOR_BGR2RGB)), plt.title('Com Ruído'), plt.axis('off')
plt.subplot(133), plt.imshow(cv.cvtColor(denoised, cv.COLOR_BGR2RGB)), plt.title('Denoised'), plt.axis('off')
plt.tight_layout()
plt.show()
