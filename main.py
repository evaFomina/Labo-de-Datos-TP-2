# -*- coding: utf-8 -*-
"""
Materia        : Laboratorio de datos - FCEyN - UBA
Contenido      : Trabajo Práctico 02 
Grupo EcuJaRu2 : Fomina, Evangelina
                 Niikado, Marina
                 Borja, Kurt 
Fecha          : 1C2024
"""
#%%===========================================================================
# Imports
#=============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, 
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score)
import random
from joblib import Parallel, delayed
#%%======== plot-letters.py  ==================================================
path = ""
data = pd.read_csv(path + "emnist_letters_tp.csv", header= None)
#%%
# Elijo la fila correspondiente a la letra que quiero graficar
n_row = 100
row = data.iloc[n_row].drop(0)
letra = data.iloc[n_row][0]

image_array = np.array(row).astype(np.float32)

# Ploteo el grafico
plt.imshow(image_array.reshape(28, 28))
plt.title('letra: ' + letra)
plt.axis('off')  
plt.show()

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
plt.imshow(flip_rotate(image_array))
plt.title('letra: ' + letra)
plt.axis('off')  
plt.show()
#%%======== FIN  plot-letters.py  =============================================

#%% ===========================================================================
# 1. Realizar un análisis exploratorio de los datos. Entre otras cosas, deben
# analizar la cantidad de datos, cantidad y tipos de atributos, cantidad de clases
# de la variable de interés (letras) y otras características que consideren
# relevantes. Además se espera que con su análisis puedan responder las
# siguientes preguntas:
#%%
cant_filas = data.count(axis=1)
cant_cols = data.count(axis=0)
clases_letras = data[0].unique()
col_letras = data[0]
cols_datos = data.drop(0, axis = 1)

#%%
# a. ¿Cuáles parecen ser atributos relevantes para predecir la letra a la que
# corresponde la imagen? ¿Cuáles no? ¿Creen que se pueden
# descartar atributos?
#%%
# b. ¿Hay letras que son parecidas entre sí? Por ejemplo, ¿Qué es más
# fácil de diferenciar: las imágenes correspondientes a la letra E de las
# correspondientes a la L, o la letra E de la M?
#%%
# c. Tomen una de las clases, por ejemplo la letra C, ¿Son todas las
# imágenes muy similares entre sí? hacer

letras_C = data[data[0] == 'C']
datos_C = letras_C.drop(0, axis=1)

for i in range(10):
    row = datos_C.iloc[i]
    letra = letras_C.iloc[i][0]
    
    image_array = np.array(row).astype(np.float32)

    plt.imshow(flip_rotate(image_array))
    plt.title('letra: ' + letra)
    plt.axis('off')  
    plt.show()

#%%
# d. Este dataset está compuesto por imágenes, esto plantea una
# diferencia frente a los datos que utilizamos en las clases (por ejemplo,
# el dataset de Titanic). ¿Creen que esto complica la exploración de los
# datos?
# Importante: las respuestas correspondientes a los puntos 1.a, 1.b y
# 1.c deben ser justificadas en base a gráficos de distinto tipo.
#%% ===========================================================================
# 2. Dada una imagen se desea responder la siguiente pregunta: ¿la imagen
# corresponde a la letra L o a la letra A?
#%%
# a. A partir del dataframe original, construir un nuevo dataframe que
# contenga sólo al subconjunto de imágenes correspondientes a las
# letras L o A.

letras_A = data[data[0] == 'A']
letras_L = data[data[0] == 'L']
letras_A_L = pd.concat([letras_A, letras_L])
#%%
# b. Sobre este subconjunto de datos, analizar cuántas muestras se tienen
# y determinar si está balanceado con respecto a las dos clases a
# predecir (la imagen es de la letra L o de la letra A).

#Los datos estan perfectamente balanceados. Hay 2400 de ejemplos de letras para cada letra
b = letras_A_L[[0]].value_counts()
#%%
# c. Separar los datos en conjuntos de train y test.
X = letras_A_L.drop(0, axis=1)
Y = letras_A_L[0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1,stratify=Y, test_size = 0.2)
#%%
#me aseguro de que está balanceada la separación
print(Y_train.value_counts())
#%%
# d. Ajustar un modelo de KNN considerando pocos atributos, por ejemplo 
# 3. Probar con distintos conjuntos de 3 atributos y comparar resultados.
# Analizar utilizando otras cantidades de atributos.
# Importante: Para evaluar los resultados de cada modelo usar el
# conjunto de test generado en el punto anterior.
# OBS: Utilicen métricas para problemas de clasificación como por
# ejemplo, exactitud.


# Evaluo el modelo de knn con k = 3 para 3 atributos seleccionados al azar

knn = KNeighborsClassifier(n_neighbors=3)
atributos = X_train.columns.tolist()
combinaciones_3 = list(combinations(atributos, 3))
num_muestras = 100
muestras_seleccionadas = random.sample(combinaciones_3, num_muestras)

def evaluar_combinacion(muestra):
    X_train_subset = X_train[list(muestra)]
    X_test_subset = X_test[list(muestra)]
 
# Evaluo 100 modelos con 3 atributos seleccionados con random
    knn.fit(X_train_subset, Y_train)
    y_pred = knn.predict(X_test_subset)
    exactitud = accuracy_score(Y_test, y_pred)
    
    return exactitud, muestra

# Funciones paralel y delayed permiten ejecutar el bucle de evaluación de combinaciones en paralelo, aprovechando todos los núcleos de CPU disponibles.
resultados = Parallel(n_jobs=-1)(delayed(evaluar_combinacion)(muestra) for muestra in muestras_seleccionadas)


# Las mejores combinaciones
resultados_ordenados = sorted(resultados, key=lambda x: x[0], reverse=True)
mejores_5 = resultados_ordenados[:5]

# Imprimo los 5 mejores conjuntos de atributos y sus exactitudes
for i, (exactitud, atributos) in enumerate(mejores_5, 1):
    print(f"Conjunto de atributos #{i}: {atributos}")
    print(f"Exactitud: {exactitud}")

#%%
# Para 5 atributos   

# Selecciono atributos mencionados en el raking del punto anterior
atributos_5 = X_train[[465, 547, 736, 548, 694]]
atributos_5_test = X_test[[465, 547, 736, 548, 694]]

knn.fit(atributos_5, Y_train)
y_pred = knn.predict(atributos_5_test)
exactitud = accuracy_score(Y_test, y_pred)
print("exactitud: ", exactitud)

# Mejora el accuracy
#%%
# Para 10 atributos
atributos_10 = X_train[[465, 547, 736, 548, 694, 765, 72, 518, 546, 493]]
atributos_10_test = X_test[[465, 547, 736, 548, 694, 765, 72, 518, 546, 493]]

knn.fit(atributos_10, Y_train)
y_pred = knn.predict(atributos_10_test)
exactitud = accuracy_score(Y_test, y_pred)
print("exactitud: ", exactitud)

# Con mas atributos mejora el accuracy
#%%
# e. Comparar modelos de KNN utilizando distintos atributos y distintos
# valores de k (vecinos). Para el análisis de los resultados, tener en
# cuenta las medidas de evaluación (por ejemplo, la exactitud) y la
# cantidad de atributos.

# Voy a usar de a 3 atributos porque en la funcion que se hizo en el punto d ya obtuve una lista de atributos.
valores_k = [3, 5, 7, 9, 12]

# Defino la función para evaluar una combinación de atributos con un valor de k
def evaluar_combinacion_k(combinacion, k):
    atributos = list(combinacion)
    X_train_subset = X_train[atributos]
    X_test_subset = X_test[atributos]
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_subset, Y_train)
    y_pred = knn.predict(X_test_subset)
    exactitud = accuracy_score(Y_test, y_pred)
    
    return exactitud, combinacion, k

# Evaluo las mejores combinaciones con diferentes valores de k
resultados = Parallel(n_jobs=-1)(
    delayed(evaluar_combinacion_k)(combinacion, k)
    for _, combinacion in mejores_5
    for k in valores_k
)

# Ordeno resultados por exactitud en orden descendente y obtengo las 10 mejores combinaciones
resultados_ordenados = sorted(resultados, key=lambda x: x[0], reverse=True)

# Imprimo los resultados para cada valor de k
resultados_por_k = {k: [] for k in valores_k}
for exactitud, combinacion, k in resultados_ordenados:
    resultados_por_k[k].append((exactitud, combinacion))

for k in valores_k:
    print(f"\nResultados para k={k}:")
    for i, (exactitud, combinacion) in enumerate(resultados_por_k[k], 1):
        print(f"Combinación #{i}: {combinacion}")
        print(f"Exactitud: {exactitud}")
############
#El mejor resultado de 3 atributos me dio:
# Con  k=9:
# Combinación #1: (254, 547, 651)
# Exactitud: 0.9604166666666667
#%% ===========================================================================
# 3. (Clasificación multiclase) Dada una imagen se desea responder la
# siguiente pregunta: ¿A cuál de las vocales corresponde la imagen?
#%%
# a. Vamos a trabajar con los datos correspondientes a las 5 vocales.
# Primero filtrar solo los datos correspondientes a esas letras. Luego,
# separar el conjunto de datos en train y test.
vocales = data[(data[0] == 'A') | 
               (data[0] == 'E') | 
               (data[0] == 'I') | 
               (data[0] == 'O') |
               (data[0] == 'U')]

X = vocales.drop(0, axis=1)
Y = vocales[0]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, 
    Y, 
    random_state = 1,
    stratify= Y, 
    test_size = 0.2)

#%%
# b. Ajustar un modelo de árbol de decisión. Analizar distintas
# profundidades.
alturas = range(1,22)

resultados_3b = np.zeros(len(alturas))

for i, hmax in enumerate(alturas):
    modelo = tree.DecisionTreeClassifier(max_depth = hmax)
    modelo.fit(X_train, Y_train)

    resultados_3b[i] = modelo.score(X_train, Y_train)
    
#%% Performance vs profundidad del modelo de árbol de decisión
plt.figure(figsize=(12, 6))
plt.plot(alturas, resultados_3b, label='Train', marker='o')
plt.xlabel('Profundidad del árbol')
plt.ylabel('Performance del modelo')
plt.legend()
plt.grid(True)
plt.show()    

#%%
# c. Para comparar y seleccionar los árboles de decisión, utilizar validación
# cruzada con k-folding.
# Importante: Para hacer k-folding utilizar los datos del conjunto de
# train.
alturas = [5,6,7,8,9,10,11,12,15,18]      
nsplits = 10
kf = KFold(n_splits=nsplits)

resultados_test_3c = np.zeros((nsplits, len(alturas)))
resultados_train_3c = np.zeros((nsplits, len(alturas)))

for i, (train_index, test_index) in enumerate(kf.split(X_train)):

    kf_X_train, kf_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
    kf_Y_train, kf_Y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
    
    for j, h in enumerate(alturas):
        
        modelo = tree.DecisionTreeClassifier(max_depth=h)
        modelo.fit(kf_X_train, kf_Y_train)
        
        resultados_test_3c[i,j] = modelo.score(kf_X_test, kf_Y_test)
        resultados_train_3c[i,j] = modelo.score(kf_X_train, kf_Y_train)

scores_promedio_test_3c = resultados_test_3c.mean(axis = 0)
scores_promedio_train_3c = resultados_train_3c.mean(axis = 0)

#%% Performance vs profundidad del modelo de árbol de decisión
plt.figure(figsize=(12, 6))
plt.plot(alturas, scores_promedio_test_3c, label='Test', marker='o')
plt.plot(alturas, scores_promedio_train_3c, label='Train', marker='o')
plt.xlabel('Profundidad del árbol')
plt.ylabel('Performance del modelo')
plt.legend()
plt.grid(True)
plt.show()    

#%%
# d. ¿Cuál fue el mejor modelo? Evaluar cada uno de los modelos
# utilizando el conjunto de test. Reportar su mejor modelo en el informe.
# OBS: Al realizar la evaluación utilizar métricas de clasificación
# multiclase. Además pueden realizar una matriz de confusión y evaluar
# los distintos tipos de errores para las clases.

# Mejor modelo: max_depth = 9
modelo = tree.DecisionTreeClassifier(max_depth=9)
modelo.fit(X_train, Y_train)

Y_pred_3d = modelo.predict(X_test)
#%% Matriz de confusión 
matriz_confusion_3d = confusion_matrix(Y_test, Y_pred_3d)
#%% Accuracy 
accuracy_3d = accuracy_score(Y_test, Y_pred_3d)
#%%  Precision 
precision_3d = precision_score(Y_test, Y_pred_3d, average='micro')
#%% Recall 
recall_3d = recall_score(Y_test, Y_pred_3d, average='micro')
#%% F1 
f1_3d = f1_score(Y_test, Y_pred_3d, average='micro')
