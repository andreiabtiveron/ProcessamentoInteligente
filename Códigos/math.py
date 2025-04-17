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
    img = cv2.imread(image_path)  # Open the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return img

def erosion(img, kernel_size=(5, 5)):
    """
    Applies erosion to the image.
    
    Parameters:
        img (numpy.ndarray): The RGB image.
        kernel_size (tuple): Size of the structuring element.
    
    Returns:
        eroded (numpy.ndarray): The eroded image.
    """
    kernel = np.ones(kernel_size, np.uint8)  # Create a structuring element
    eroded = cv2.erode(img, kernel, iterations=1)  # Apply erosion
    return eroded

def dilation(img, kernel_size=(5, 5)):
    """
    Applies dilation to the image.
    
    Parameters:
        img (numpy.ndarray): The RGB image.
        kernel_size (tuple): Size of the structuring element.
    
    Returns:
        dilated (numpy.ndarray): The dilated image.
    """
    kernel = np.ones(kernel_size, np.uint8)  # Create a structuring element
    dilated = cv2.dilate(img, kernel, iterations=1)  # Apply dilation
    return dilated

def opening(img, kernel_size=(5, 5)):
    """
    Applies morphological opening to the image.
    
    Parameters:
        img (numpy.ndarray): The RGB image.
        kernel_size (tuple): Size of the structuring element.
    
    Returns:
        opened (numpy.ndarray): The opened image.
    """
    kernel = np.ones(kernel_size, np.uint8)  # Create a structuring element
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Apply opening
    return opened

def closing(img, kernel_size=(5, 5)):
    """
    Applies morphological closing to the image.
    
    Parameters:
        img (numpy.ndarray): The RGB image.
        kernel_size (tuple): Size of the structuring element.
    
    Returns:
        closed (numpy.ndarray): The closed image.
    """
    kernel = np.ones(kernel_size, np.uint8)  # Create a structuring element
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # Apply closing
    return closed

def display_images(original, eroded, dilated, opened, closed):
    """
    Displays the original and processed images using matplotlib.
    
    Parameters:
        original (numpy.ndarray): The original image.
        eroded (numpy.ndarray): The eroded image.
        dilated (numpy.ndarray): The dilated image.
        opened (numpy.ndarray): The opened image.
        closed (numpy.ndarray): The closed image.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(eroded)
    plt.title('Eroded Image')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(dilated)
    plt.title('Dilated Image')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(opened)
    plt.title('Opened Image')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(closed)
    plt.title('Closed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
image_path = 'correcaoB.jpg'  # Replace with your image path
rgb_image = open_image(image_path)
eroded_image = erosion(rgb_image)
dilated_image = dilation(rgb_image)
opened_image = opening(rgb_image)
closed_image = closing(rgb_image)

# Display all images
display_images(rgb_image, eroded_image, dilated_image, opened_image, closed_image)