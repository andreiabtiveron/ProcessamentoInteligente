import cv2
import numpy as np
import matplotlib.pyplot as plt

def open_image(image_path):
    """
    Opens an RGB image from the specified path and converts to grayscale.
    
    Parameters:
        image_path (str): Path to the image file.
    
    Returns:
        img (numpy.ndarray): The opened RGB image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def binarize_image(img, threshold=128):
    """
    Converts the RGB image to a binary image using a specified threshold.
    
    Parameters:
        img (numpy.ndarray): The RGB image.
        threshold (int): The threshold value for binarization.
    
    Returns:
        binary_img (numpy.ndarray): The binary image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    _, binary_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)  # Binarize
    return binary_img

def hit_and_miss(binary_img, kernel):
    """
    Applies the Hit-and-Miss transformation.
    
    Parameters:
        binary_img (numpy.ndarray): The binary image.
        kernel (numpy.ndarray): The structuring element for the Hit-and-Miss transformation.
    
    Returns:
        hit_and_miss_result (numpy.ndarray): The result of the Hit-and-Miss transformation.
    """
    # Create a mask for the inverse of the kernel
    kernel_inv = cv2.bitwise_not(kernel)
    
    # Erode with the kernel and the inverse kernel
    eroded_with_kernel = cv2.erode(binary_img, kernel)
    eroded_with_inv_kernel = cv2.erode(255 - binary_img, kernel_inv)
    
    # Combine results: logical AND operation
    hit_and_miss_result = cv2.bitwise_and(eroded_with_kernel, eroded_with_inv_kernel)
    
    return hit_and_miss_result

# === USO DO SCRIPT ===

image_path = 'bordaRealceB1.jpg'  # Caminho da imagem
rgb_image = open_image(image_path)
binary_image = binarize_image(rgb_image)

# Define um elemento estruturante (cruz 3x3)
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)

# Aplica Hit-and-Miss (caso ainda queira usar o resultado depois)
hit_miss_result = hit_and_miss(binary_image, kernel)

# Mostra só a imagem binária
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')
plt.show()

# Salva a imagem binária
cv2.imwrite('imagem_binaria.png', binary_image)
