import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MeanShift

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

def dbscan_clustering(feature_vectors, eps, min_samples):
    """
    Applies DBSCAN clustering on the feature vectors.
    
    Parameters:
        feature_vectors (np.ndarray): An array of feature vectors.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
        labels (np.ndarray): The labels of each point after clustering.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(feature_vectors)
    return labels

def agglomerative_clustering(feature_vectors, n_clusters):
    """
    Applies Agglomerative Clustering on the feature vectors.
    
    Parameters:
        feature_vectors (np.ndarray): An array of feature vectors.
        n_clusters (int): The number of clusters to form.
    
    Returns:
        labels (np.ndarray): The labels of each point after clustering.
    """
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglomerative.fit_predict(feature_vectors)
    return labels

def meanshift_clustering(feature_vectors):
    """
    Applies Mean Shift clustering on the feature vectors.
    
    Parameters:
        feature_vectors (np.ndarray): An array of feature vectors.
    
    Returns:
        labels (np.ndarray): The labels of each point after clustering.
    """
    meanshift = MeanShift()
    labels = meanshift.fit_predict(feature_vectors)
    return labels

def display_results(labels, num_layers, method):
    """
    Displays the clustering results.
    
    Parameters:
        labels (np.ndarray): The labels of each point after clustering.
        num_layers (int): The number of layers in the image stack.
        method (str): The clustering method used.
    """
    print(f"Clustering Results using {method}:")
    for i in range(num_layers):
        print(f"Layer {i + 1} is in Cluster {labels[i]}")

# Example usage
num_layers = int(input("How many layers will your image stack have? "))

# Create image stack
image_stack = create_image_stack(num_layers)

# Compute feature vectors
feature_vectors = compute_feature_vectors(image_stack)

# Choose clustering method
print("Choose a clustering method:")
print("1. DBSCAN")
print("2. Agglomerative Clustering")
print("3. Mean Shift")
method_choice = int(input("Enter the number of the chosen method (1-3): "))

if method_choice == 1:
    eps = float(input("Enter the value for eps (e.g., 0.5): "))
    min_samples = int(input("Enter the minimum samples (e.g., 2): "))
    labels = dbscan_clustering(feature_vectors, eps, min_samples)
    display_results(labels, num_layers, "DBSCAN")
elif method_choice == 2:
    n_clusters = int(input("Enter the number of clusters for Agglomerative Clustering: "))
    labels = agglomerative_clustering(feature_vectors, n_clusters)
    display_results(labels, num_layers, "Agglomerative Clustering")
elif method_choice == 3:
    labels = meanshift_clustering(feature_vectors)
    display_results(labels, num_layers, "Mean Shift")
else:
    print("Invalid choice! Please choose a valid clustering method.")