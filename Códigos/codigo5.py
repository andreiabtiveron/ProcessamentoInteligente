import cv2
import numpy as np
import matplotlib.pyplot as plt

def open_image(image_path):
    """
    Opens an RGB image from the specified path.
    
    Parameters:
        image_path (str): Path to the image file.
    
    Returns:
        img (numpy.ndarray): The opened RGB image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def apply_low_pass_filter(img, kernel):
    """
    Applies a low-pass filter using the specified kernel.
    
    Parameters:
        img (numpy.ndarray): The RGB image.
        kernel (numpy.ndarray): The low-pass filter kernel.
    
    Returns:
        low_pass_img (numpy.ndarray): The filtered image.
    """
    # Apply convolution
    filtered = cv2.filter2D(img, -1, kernel)
    return filtered

def display_images(original, filtered_images):
    """
    Displays the original and filtered images using matplotlib.
    
    Parameters:
        original (numpy.ndarray): The original image.
        filtered_images (list): List of filtered images.
    """
    plt.figure(figsize=(12, 6))
    
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

# Low-pass filter kernels
kernels = [
    np.array([[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]], dtype=np.float32) / 9,  # Average filter

    np.array([[1, 2, 1],
               [2, 4, 2],
               [1, 2, 1]], dtype=np.float32) / 16,  # Gaussian filter

    np.array([[1, 1, 1],
               [1, 2, 1],
               [1, 1, 1]], dtype=np.float32) / 10,  # Box filter

    np.array([[1, 2, 1],
               [2, 6, 2],
               [1, 2, 1]], dtype=np.float32) / 16   # Weighted average filter
]

# Example usage
image_path = 'Figura3.jpeg'  # Replace with your image path
rgb_image = open_image(image_path)

# Apply low-pass filters
filtered_images = [apply_low_pass_filter(rgb_image, kernel) for kernel in kernels]

# Display all images
display_images(rgb_image, filtered_images)