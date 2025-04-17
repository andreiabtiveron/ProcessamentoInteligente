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

def display_images(original, binary_img, hit_and_miss_result):
    """
    Displays the original, binary, and Hit-and-Miss images using matplotlib.
    
    Parameters:
        original (numpy.ndarray): The original image.
        binary_img (numpy.ndarray): The binary image.
        hit_and_miss_result (numpy.ndarray): The result of the Hit-and-Miss transformation.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hit_and_miss_result, cmap='gray')
    plt.title('Hit and Miss Result')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
image_path = 'correcaoB.jpg'  # Replace with your image path
rgb_image = open_image(image_path)
binary_image = binarize_image(rgb_image)

# Define a simple structuring element (for example, a 3x3 cross)
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)

# Apply Hit-and-Miss transformation
hit_miss_result = hit_and_miss(binary_image, kernel)

# Display all images
display_images(rgb_image, binary_image, hit_miss_result)