#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:39:20 2024

@author: docentes Labo Datos
"""
# script para plotear letras


# import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
#%%
# data = pd.read_csv("emnist_letters_tp.csv", header= None)
#%%
# Elijo la fila correspondiente a la letra que quiero graficar
# n_row = 100
# row = data.iloc[n_row].drop(0)
# letra = data.iloc[n_row][0]

# image_array = np.array(row).astype(np.float32)

# Ploteo el grafico
# plt.imshow(image_array.reshape(28, 28))
# plt.title('letra: ' + letra)
# plt.axis('off')  
# plt.show()

# Se observa que las letras estan rotadas en 90° y espejadas
#%%
def flip_rotate(image):
    """
    Función que recibe un array de numpy representando una
    imagen de 28x28. Espeja el array y lo rota en 90°.
    """
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# Ploteo la imagen transformada
# plt.imshow(flip_rotate(image_array))
# plt.title('letra: ' + letra)
# plt.axis('off')  
# plt.show()

