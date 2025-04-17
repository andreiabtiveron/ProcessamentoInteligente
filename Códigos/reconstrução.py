import cv2
import numpy as np
from sklearn.decomposition import PCA
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

def create_image_stack(num_layers):
    """
    Creates an image stack based on user input for the number of layers.
    
    Parameters:
        num_layers (int): The number of layers in the image stack.
    
    Returns:
        image_stack (list): A list containing the images stacked.
    """
    image_stack = []
    
    for i in range(num_layers):
        image_path = input(f"Enter the path for image layer {i + 1}: ")
        img = open_image(image_path)
        image_stack.append(img)
        
    return image_stack

def compute_feature_vectors(image_stack):
    """
    Computes feature vectors from the image stack.
    
    Parameters:
        image_stack (list): A list of images in the stack.
    
    Returns:
        feature_vectors (np.ndarray): An array of feature vectors for each image.
    """
    feature_vectors = []
    
    for img in image_stack:
        # Calculate mean color values as a feature vector
        mean_color = cv2.mean(img)[:3]  # Get the mean for B, G, R channels
        feature_vectors.append(mean_color)
        
    return np.array(feature_vectors)

def apply_pca(feature_vectors, n_components):
    """
    Applies PCA to reduce the dimensionality of the feature vectors.
    
    Parameters:
        feature_vectors (np.ndarray): An array of feature vectors.
        n_components (int): The number of principal components to keep.
    
    Returns:
        pca_result (np.ndarray): The transformed feature vectors in PCA space.
        pca: The fitted PCA model.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(feature_vectors)
    return pca_result, pca

def reconstruct_images(image_stack, pca, num_components):
    """
    Reconstructs the images using the selected principal components.
    
    Parameters:
        image_stack (list): The original image stack.
        pca: The fitted PCA model.
        num_components (int): The number of components to use for reconstruction.
    
    Returns:
        reconstructed_images (list): A list of reconstructed RGB images.
    """
    reconstructed_images = []
    
    for img in image_stack:
        img_flattened = img.reshape(-1, 3)  # Flatten the image to 2D array
        img_pca = pca.transform(img_flattened)[:, :num_components]  # Project to PCA space
        img_reconstructed = pca.inverse_transform(img_pca)  # Inverse transform to reconstruct
        img_reconstructed = np.clip(img_reconstructed, 0, 255).astype(np.uint8)  # Clip to valid range
        reconstructed_images.append(img_reconstructed.reshape(img.shape))  # Reshape back to original dimensions
    
    return reconstructed_images

def display_images(original, reconstructed):
    """
    Displays original and reconstructed images.
    
    Parameters:
        original (list): List of original images.
        reconstructed (list): List of reconstructed images.
    """
    plt.figure(figsize=(12, 6))

    for i in range(len(original)):
        plt.subplot(2, len(original), i + 1)
        plt.imshow(original[i])
        plt.title(f'Original Layer {i + 1}')
        plt.axis('off')

        plt.subplot(2, len(original), i + 1 + len(original))
        plt.imshow(reconstructed[i])
        plt.title(f'Reconstructed Layer {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
num_layers = int(input("How many layers will your image stack have? "))
image_stack = create_image_stack(num_layers)

# Compute feature vectors
feature_vectors = compute_feature_vectors(image_stack)

# Get number of components for PCA from user
n_components = int(input("Enter the number of principal components to keep (1-3): "))

# Apply PCA
pca_result, pca = apply_pca(feature_vectors, n_components)

# Reconstruct images
reconstructed_images = reconstruct_images(image_stack, pca, n_components)

# Display original and reconstructed images
display_images(image_stack, reconstructed_images)