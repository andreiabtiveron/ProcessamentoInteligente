import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

def get_labels(num_layers):
    """
    Gets labels for each layer from the user.
    
    Parameters:
        num_layers (int): The number of layers in the image stack.
    
    Returns:
        labels (list): A list of labels corresponding to each layer.
    """
    labels = []
    for i in range(num_layers):
        label = input(f"Enter the label for layer {i + 1}: ")
        labels.append(label)
    return labels

def train_svm_classifier(X, y):
    """
    Trains an SVM classifier on the feature vectors and labels.
    
    Parameters:
        X (np.ndarray): Feature vectors.
        y (list): Corresponding labels.
    
    Returns:
        model: The trained SVM model.
    """
    model = svm.SVC(kernel='linear')  # Using a linear kernel
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the SVM model on test data.
    
    Parameters:
        model: The trained SVM model.
        X_test (np.ndarray): Test feature vectors.
        y_test (list): Test labels.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Example usage
num_layers = int(input("How many layers will your image stack have? "))

# Create image stack
image_stack = create_image_stack(num_layers)

# Compute feature vectors
feature_vectors = compute_feature_vectors(image_stack)

# Get labels for each layer
labels = get_labels(num_layers)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)

# Train SVM classifier
model = train_svm_classifier(X_train, y_train)

# Evaluate the model
evaluate_model(model, X_test, y_test)