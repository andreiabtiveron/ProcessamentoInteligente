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

def apply_canny_edge_detection(img, low_threshold=100, high_threshold=200):
    """
    Applies Canny edge detection to the image.
    
    Parameters:
        img (numpy.ndarray): The RGB image.
        low_threshold (int): Lower threshold for the hysteresis procedure.
        high_threshold (int): Upper threshold for the hysteresis procedure.
    
    Returns:
        edges (numpy.ndarray): The image with Canny edges detected.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def display_images(original, edges):
    """
    Displays the original image and the edges detected image using matplotlib.
    
    Parameters:
        original (numpy.ndarray): The original image.
        edges (numpy.ndarray): The image with Canny edges detected.
    """
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
image_path = 'path/to/your/image.jpg'  # Replace with your image path
rgb_image = open_image(image_path)

# Apply Canny edge detection
edges_image = apply_canny_edge_detection(rgb_image)

# Display original and edges images
display_images(rgb_image, edges_image)