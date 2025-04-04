import cv2
import numpy as np 
import matplotlib.pyplot as plt


image_path = "aula007.png"  
image = cv2.imread(image_path)


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])


mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)


result_blue = cv2.bitwise_and(image, image, mask=mask_blue)


plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_blue, cmap="gray")
plt.title("Máscara de Azul")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result_blue, cv2.COLOR_BGR2RGB))
plt.title("Detecção Completa do Azul")
plt.axis("off")

plt.show()
