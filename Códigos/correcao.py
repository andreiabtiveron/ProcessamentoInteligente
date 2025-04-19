import cv2
import numpy as np
import matplotlib.pyplot as plt

def correct_blue_tint(image):
    # Converter de RGB para LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Separar canais
    l, a, b = cv2.split(lab)

    # Equalizar histograma do canal de luminância (L)
    l_eq = cv2.equalizeHist(l)

    # Recompor a imagem LAB
    lab_eq = cv2.merge((l_eq, a, b))

    # Converter de volta para RGB
    corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return corrected

def white_balance_simple(image):
    # Converter para float32
    img_float = image.astype(np.float32)

    # Calcular a média de cada canal
    avg_b = np.mean(img_float[:,:,2])
    avg_g = np.mean(img_float[:,:,1])
    avg_r = np.mean(img_float[:,:,0])

    # Calcular fator de correção
    avg = (avg_b + avg_g + avg_r) / 3
    img_float[:,:,2] *= avg / avg_b
    img_float[:,:,1] *= avg / avg_g
    img_float[:,:,0] *= avg / avg_r

    # Clipping e conversão para uint8
    balanced = np.clip(img_float, 0, 255).astype(np.uint8)
    return balanced

# Abrir a imagem
image_path = 'correcaoB.jpg'
img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Corrigir azul e melhorar histograma
corrected = correct_blue_tint(img_rgb)

# Aplicar balanço de branco simples
balanced = white_balance_simple(corrected)

# Mostrar resultado
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title('Original com Tom Azul')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(corrected)
plt.title('Após Equalização LAB')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(balanced)
plt.title('Após Balanço de Branco')
plt.axis('off')

plt.tight_layout()
plt.show()
