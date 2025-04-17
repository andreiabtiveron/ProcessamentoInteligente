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

def preprocess_image(img):
    """
    Preprocess the image for contour detection: convert to grayscale and apply thresholding.
    
    Parameters:
        img (numpy.ndarray): The RGB image.
    
    Returns:
        thresh (numpy.ndarray): The binary thresholded image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

def extract_shape_parameters(contour):
    """
    Extracts shape parameters from a given contour.
    
    Parameters:
        contour (numpy.ndarray): A contour of an object.
    
    Returns:
        parameters (dict): A dictionary containing shape parameters.
    """
    # Calculate area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate circularity
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0
    
    # Calculate bounding rectangle and aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    # Calculate convex hull and convexity
    hull = cv2.convexHull(contour)
    convexity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) != 0 else 0
    
    # Calculate solidity
    solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) != 0 else 0
    
    # Calculate equivalent diameter
    equivalent_diameter = np.sqrt(4 * area / np.pi) if area != 0 else 0
    
    # Calculate elongation
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    elongation = MA / ma if ma != 0 else 0
    
    # Calculate rectangularity
    rectangularity = area / (w * h) if w * h != 0 else 0
    
    # Calculate Feret diameter
    feret_diameter = max(cv2.minEnclosingCircle(contour)[1], 0)  # Circle radius as diameter
    
    parameters = {
        'Area': area,
        'Perimeter': perimeter,
        'Circularity': circularity,
        'Aspect Ratio': aspect_ratio,
        'Convexity': convexity,
        'Solidity': solidity,
        'Equivalent Diameter': equivalent_diameter,
        'Elongation': elongation,
        'Rectangularity': rectangularity,
        'Feret Diameter': feret_diameter
    }
    
    return parameters

def display_results(original, contours):
    """
    Displays the original image with contours and prints shape parameters.
    
    Parameters:
        original (numpy.ndarray): The original image.
        contours (list): List of contours.
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(original)
    plt.title('Contours')
    plt.axis('off')
    
    for contour in contours:
        params = extract_shape_parameters(contour)
        print(params)
    
    plt.show()

# Example usage
image_path = 'path/to/your/image.jpg'  # Replace with your image path
rgb_image = open_image(image_path)

# Preprocess the image
thresh_image = preprocess_image(rgb_image)

# Find contours
contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contour_image = rgb_image.copy()
cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)

# Display results
display_results(contour_image, contours)