import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
        # Calculate mean color values as a simple feature vector
        mean_color = cv2.mean(img)[:3]  # Get the mean for B, G, R channels
        feature_vectors.append(mean_color)
        
    return np.array(feature_vectors)

def k_means_clustering(feature_vectors, n_clusters):
    """
    Applies K-means clustering on the feature vectors.
    
    Parameters:
        feature_vectors (np.ndarray): An array of feature vectors.
        n_clusters (int): The number of clusters to form.
    
    Returns:
        labels (np.ndarray): The labels of each point after clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(feature_vectors)
    return kmeans.labels_

def display_results(labels, num_layers):
    """
    Displays the clustering results.
    
    Parameters:
        labels (np.ndarray): The labels of each point after clustering.
        num_layers (int): The number of layers in the image stack.
    """
    print("Clustering Results:")
    for i in range(num_layers):
        print(f"Layer {i + 1} is in Cluster {labels[i]}")

# Example usage
num_layers = int(input("How many layers will your image stack have? "))

# Create image stack
image_stack = create_image_stack(num_layers)

# Compute feature vectors
feature_vectors = compute_feature_vectors(image_stack)

# Get number of clusters from user
n_clusters = int(input("Enter the number of clusters for K-means: "))

# Perform K-means clustering
labels = k_means_clustering(feature_vectors, n_clusters)

# Display results
display_results(labels, num_layers)