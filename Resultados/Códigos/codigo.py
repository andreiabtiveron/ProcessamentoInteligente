# importing required libraries 
import mahotas as mh 
import numpy as np 
from pylab import imshow, show 

# carregando a imagem classifB (substitua pelo caminho correto se necessário)
classifB = mh.imread('classifB.jpg')  # ou .jpg, .tif, etc.

# se a imagem for colorida, converte para escala de cinza
if classifB.ndim == 3:
    classifB = mh.colors.rgb2gray(classifB)

# binarização (ajuste o limiar conforme necessário)
threshold = mh.thresholding.otsu(classifB)
binary = classifB > threshold

# rotulando as regiões
labeled, nr_objects = mh.label(binary)

# exibindo a imagem rotulada
print("Imagem rotulada (classifB)")
imshow(labeled, interpolation='nearest')
show()

# detectando bordas com DoG
dog = mh.dog(labeled.astype(np.uint8))

# exibindo bordas
print("Bordas usando o algoritmo DoG")
imshow(dog)
show()