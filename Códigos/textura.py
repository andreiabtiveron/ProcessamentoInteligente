import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, measure

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

def compute_glcm(gray_image, distances=[1], angles=[0]):
    """
    Computes the Gray-Level Co-occurrence Matrix (GLCM).
    
    Parameters:
        gray_image (numpy.ndarray): The grayscale image.
        distances (list): List of pixel pair distance(s).
        angles (list): List of angles (in radians) for the GLCM.
    
    Returns:
        glcm (numpy.ndarray): The computed GLCM.
    """
    glcm = feature.greycomatrix(gray_image, distances, angles, symmetric=True, normed=True)
    return glcm

def calculate_entropy(glcm):
    """
    Calculates the entropy of the GLCM.
    
    Parameters:
        glcm (numpy.ndarray): The computed GLCM.
    
    Returns:
        entropy (float): The entropy value.
    """
    # Sum the GLCM across all angles and distances
    glcm_sum = np.sum(glcm, axis=2)
    entropy = -np.sum(glcm_sum[glcm_sum > 0] * np.log(glcm_sum[glcm_sum > 0]))
    return entropy

def extract_glcm_features(glcm):
    """
    Extracts texture features from the GLCM.
    
    Parameters:
        glcm (numpy.ndarray): The computed GLCM.
    
    Returns:
        features (dict): A dictionary containing the texture features.
    """
    features = {}
    
    # Extract features for each angle and distance
    for i in range(glcm.shape[2]):
        features['contrast'] = feature.greycoprops(glcm, 'contrast')[0][i]
        features['dissimilarity'] = feature.greycoprops(glcm, 'dissimilarity')[0][i]
        features['homogeneity'] = feature.greycoprops(glcm, 'homogeneity')[0][i]
        features['energy'] = feature.greycoprops(glcm, 'energy')[0][i]
        features['correlation'] = feature.greycoprops(glcm, 'correlation')[0][i]
        features['entropy'] = calculate_entropy(glcm)
    
    return features

def display_results(original, gray_image, glcm_features):
    """
    Displays the original image and grayscale image, and prints the GLCM features.
    
    Parameters:
        original (numpy.ndarray): The original image.
        gray_image (numpy.ndarray): The grayscale image.
        glcm_features (dict): The extracted GLCM features.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("GLCM Features:")
    for feature_name, value in glcm_features.items():
        print(f"{feature_name}: {value:.4f}")

# Example usage
image_path = 'path/to/your/image.jpg'  # Replace with your image path
rgb_image = open_image(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

# Compute GLCM
glcm = compute_glcm(gray_image)

# Extract features
glcm_features = extract_glcm_features(glcm)

# Display results
display_results(rgb_image, gray_image, glcm_features)