import cv2
import numpy as np
import matplotlib.pyplot as plt

def open_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def split_rgb_channels(img):
    r_channel, g_channel, b_channel = cv2.split(img)
    return r_channel, g_channel, b_channel

def display_colored_channels(r_channel, g_channel, b_channel):
    zeros = np.zeros_like(r_channel)

    red_img = np.stack((r_channel, zeros, zeros), axis=2)
    green_img = np.stack((zeros, g_channel, zeros), axis=2)
    blue_img = np.stack((zeros, zeros, b_channel), axis=2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(red_img)
    plt.title('Red Channel')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(green_img)
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blue_img)
    plt.title('Blue Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def merge_rgb_channels(r_channel, g_channel, b_channel):
    return cv2.merge((r_channel, g_channel, b_channel))

# === USO ===
image_path = 'correcaoB.jpg'  # Ou seu caminho completo se necessário
rgb_image = open_image(image_path)
r, g, b = split_rgb_channels(rgb_image)

# Mostrar os canais
display_colored_channels(r, g, b)

# Reconstruir imagem
reconstructed = merge_rgb_channels(r, g, b)

# Mostrar imagem reconstruída
plt.figure(figsize=(6, 6))
plt.imshow(reconstructed)
plt.title("Imagem Reconstruída")
plt.axis("off")
plt.show()
