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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converte de BGR (blue green red) para RGB (red green blu)
    return img                                  # pq as imagens em OpenCV leem as imagens em BGR e o matplotlib leem no formato RGB
                                                # # Sem essa conversão:
                                                # Se você carregar uma imagem com cv2.imread() e não converter para RGB, ela será exibida com cores erradas (distorcidas) no matplotlib.pyplot.imshow() — por exemplo:
                                                # Céu azul pode parecer meio avermelhado.
                                                # Áreas verdes podem parecer roxas, etc.



def split_rgb_channels(img):
    """
    Splits the RGB channels of the given image.
    
    Parameters:
        img (numpy.ndarray): The RGB image.
    
    Returns:
        r_channel (numpy.ndarray): The red channel of the image.
        g_channel (numpy.ndarray): The green channel of the image.
        b_channel (numpy.ndarray): The blue channel of the image.
    """
    r_channel, g_channel, b_channel = cv2.split(img)  # Split into RGB channels
    return r_channel, g_channel, b_channel

def display_channels(r_channel, g_channel, b_channel): # está dividindo a imagem em um canal de 3 cores, vermelho verde e azul
    """
    Displays the RGB channels using matplotlib.
    
    Parameters:
        r_channel (numpy.ndarray): The red channel.
        g_channel (numpy.ndarray): The green channel.
        b_channel (numpy.ndarray): The blue channel.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(r_channel, cmap='gray')
    plt.title('Red Channel')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(g_channel, cmap='gray')
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(b_channel, cmap='gray')
    plt.title('Blue Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
image_path = 'correcaoB.jpg'  # Replace with your image path
rgb_image = open_image(image_path)
r, g, b = split_rgb_channels(rgb_image) # Separa a imagem RGB nos seus 3 canais de cor usando split_rgb_channels.
display_channels(r, g, b) # mostra as imagens