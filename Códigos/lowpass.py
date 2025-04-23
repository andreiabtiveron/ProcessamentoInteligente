import cv2
import numpy as np
import matplotlib.pyplot as plt

def open_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def apply_low_pass_filter(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def apply_gaussian_blur(img, ksize=(5, 5), sigma=1.5):
    return cv2.GaussianBlur(img, ksize, sigmaX=sigma)

def display_images(original, filtered_images):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, len(filtered_images) + 1, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    for i, img in enumerate(filtered_images):
        plt.subplot(1, len(filtered_images) + 1, i + 2)
        plt.imshow(img)
        plt.title(f'Filter {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Improved Low-pass filter kernels
kernels = [
    np.ones((3, 3), dtype=np.float32) / 9,  # Average 3x3
    np.array([[1, 2, 1],
              [2, 4, 2],
              [1, 2, 1]], dtype=np.float32) / 16,  # Gaussian 3x3
    np.ones((5, 5), dtype=np.float32) / 25,  # Average 5x5
    np.ones((7, 7), dtype=np.float32) / 49,  # Average 7x7
    np.array([[1, 4, 6, 4, 1],
              [4, 16, 24, 16, 4],
              [6, 24, 36, 24, 6],
              [4, 16, 24, 16, 4],
              [1, 4, 6, 4, 1]], dtype=np.float32) / 256  # Gaussian-like 5x5
]

# Load image
image_path = 'Figura_ajustada.png'
rgb_image = open_image(image_path)

# Apply low-pass filters
filtered_images = [apply_low_pass_filter(rgb_image, kernel) for kernel in kernels]

# Optional: Add GaussianBlur with parameters
gaussian_img = apply_gaussian_blur(rgb_image, ksize=(7, 7), sigma=2.0)
filtered_images.append(gaussian_img)

# Display results
display_images(rgb_image, filtered_images)
