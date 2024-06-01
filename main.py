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
from plot_letters import flip_rotate

data = pd.read_csv("emnist_letters_tp.csv")

#%%============================================================================
# 1. Realizar un análisis exploratorio de los datos. Entre otras cosas, deben
# analizar la cantidad de datos, cantidad y tipos de atributos, cantidad de clases
# de la variable de interés (letras) y otras características que consideren
# relevantes. Además se espera que con su análisis puedan responder las
# siguientes preguntas:
#==============================================================================

cant_filas = data.count(axis=1)
cant_cols = data.count(axis=0)
clases_letras = data[0].unique()
col_letras = data[0]
cols_datos = data.drop(0, axis = 1)

#%%------------------------------------------------------
# a. ¿Cuáles parecen ser atributos relevantes para predecir la letra a la que
# corresponde la imagen? ¿Cuáles no? ¿Creen que se pueden
# descartar atributos?
#--------------------------------------------------------

#%%------------------------------------------------------
# b. ¿Hay letras que son parecidas entre sí? Por ejemplo, ¿Qué es más
# fácil de diferenciar: las imágenes correspondientes a la letra E de las
# correspondientes a la L, o la letra E de la M?
#--------------------------------------------------------

#%%------------------------------------------------------
# c. Tomen una de las clases, por ejemplo la letra C, ¿Son todas las
# imágenes muy similares entre sí? hacer
#--------------------------------------------------------

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

#%%------------------------------------------------------
# d. Este dataset está compuesto por imágenes, esto plantea una
# diferencia frente a los datos que utilizamos en las clases (por ejemplo,
# el dataset de Titanic). ¿Creen que esto complica la exploración de los
# datos?
#--------------------------------------------------------

#%%============================================================================
# 2.(Clasificación binaria) Dada una imagen se desea responder la siguiente
# pregunta: ¿la imagen corresponde a la letra L o a la letra A?
#==============================================================================

#--------------------------------------------------------
# a. A partir del dataframe original, construir un nuevo dataframe que
# contenga sólo al subconjunto de imágenes correspondientes a las
# letras L o A.
#--------------------------------------------------------

letras_A = data[data[0] == 'A']
letras_L = data[data[0] == 'L']
letras_A_L = pd.concat([letras_A, letras_L])

#%%------------------------------------------------------
# b. Sobre este subconjunto de datos, analizar cuántas muestras se tienen
# y determinar si está balanceado con respecto a las dos clases a
# predecir (la imagen es de la letra L o de la letra A).
#--------------------------------------------------------

#Los datos estan perfectamente balanceados. Hay 2400 de ejemplos de letras para cada letra
b = letras_A_L[[0]].value_counts()

#%%------------------------------------------------------
# c. Separar los datos en conjuntos de train y test.
#--------------------------------------------------------

X = letras_A_L.drop(0, axis=1)
Y = letras_A_L[0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1,stratify=Y, test_size = 0.2)

#%%------------------------------------------------------
# d. Ajustar un modelo de KNN en los datos de train, considerando pocos
# atributos, por ejemplo 3. Probar con distintos conjuntos de 3 atributos y
# comparar resultados. Analizar utilizando otras cantidades de atributos.
# Para comparar los resultados de cada modelo usar el conjunto de test
# generado en el punto anterior.
# OBS: Utilicen métricas para problemas de clasificación como por
# ejemplo, exactitud.
#--------------------------------------------------------

# Evaluo el modelo de knn con k = 3 para 3 atributos seleccionados al azar

knn = KNeighborsClassifier(n_neighbors=3)
atributos = X_train.columns.tolist()
num_muestras = 100

# Función para evaluar una combinación de atributos
def evaluar_combinacion(muestra):
    X_train_subset = X_train[list(muestra)]
    X_test_subset = X_test[list(muestra)]
    
    knn.fit(X_train_subset, Y_train)
    y_pred = knn.predict(X_test_subset)
    exactitud = accuracy_score(Y_test, y_pred)
    
    return exactitud, muestra

# Genero 100 muestras aleatorias de 3 atributos
muestras_seleccionadas_3 = [random.sample(atributos, 3) for _ in range(num_muestras)]

# Evaluo las combinaciones en paralelo. Funciones paralel y delayed permiten ejecutar el bucle de
# evaluación de combinaciones en paralelo, aprovechando todos los núcleos de CPU disponibles.
# Para eso tuve que importar Parallel, delayed de joblib
resultados = Parallel(n_jobs=-1)(delayed(evaluar_combinacion)(muestra) for muestra in muestras_seleccionadas_3)

# Ordeno resultados por exactitud en orden descendente y obtengo las 5 mejores combinaciones
resultados_ordenados = sorted(resultados, key=lambda x: x[0], reverse=True)
mejores_5 = resultados_ordenados[:5]

# Imprimo los 5 mejores conjuntos de atributos y sus exactitudes
for i, (exactitud, atributos) in enumerate(mejores_5, 1):
    print(f"Conjunto de atributos #{i}: {atributos}")
    print(f"Exactitud: {exactitud}")
#%%
# Para 5 atributos   

atributos = X_train.columns.tolist()
num_muestras = 100

muestras_seleccionadas_5 = [random.sample(atributos, 5) for _ in range(num_muestras)]

resultados = Parallel(n_jobs=-1)(delayed(evaluar_combinacion)(muestra) for muestra in muestras_seleccionadas_5)


resultados_ordenados = sorted(resultados, key=lambda x: x[0], reverse=True)
mejores_5 = resultados_ordenados[:5]

for i, (exactitud, atributos) in enumerate(mejores_5, 1):
    print(f"Conjunto de atributos #{i}: {atributos}")
    print(f"Exactitud: {exactitud}")
    
# Se puede ver que el mejor accuracy es mas alto que en el de 3 atributos.
#%%
# Para 10 atributos
atributos = X_train.columns.tolist()
num_muestras = 100

muestras_seleccionadas_10 = [random.sample(atributos, 10) for _ in range(num_muestras)]

resultados = Parallel(n_jobs=-1)(delayed(evaluar_combinacion)(muestra) for muestra in muestras_seleccionadas_10)


resultados_ordenados = sorted(resultados, key=lambda x: x[0], reverse=True)
mejores_5 = resultados_ordenados[:5]

for i, (exactitud, atributos) in enumerate(mejores_5, 1):
    print(f"Conjunto de atributos #{i}: {atributos}")
    print(f"Exactitud: {exactitud}")

# Con mas atributos mejora el accuracy

#%%------------------------------------------------------
# e. Comparar modelos de KNN utilizando distintos atributos y distintos
# valores de k (vecinos). Para el análisis de los resultados, tener en
# cuenta las medidas de evaluación (por ejemplo, la exactitud) y la
# cantidad de atributos.
# Observación: en este ejercicio 2 no estamos usando k-folding ni
# estamos dejando un conjunto held-out. Solamente entrenamos en train
# y evaluamos en test, donde train y test están fijos a lo largo de los
# incisos c,d,e.
#--------------------------------------------------------


# Voy a usar de a 10 atributos ya que mejora mucho el accuracy
valores_k = [3, 5, 7, 9, 12]

def evaluar_combinacion_k(combinacion, k):
    atributos = list(combinacion)
    X_train_subset = X_train[atributos]
    X_test_subset = X_test[atributos]
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_subset, Y_train)
    y_pred = knn.predict(X_test_subset)
    exactitud = accuracy_score(Y_test, y_pred)
    
    return exactitud, combinacion, k

# Evaluo las mejores combinaciones con diferentes valores de k en paralelo
resultados_k = Parallel(n_jobs=-1)(
    delayed(evaluar_combinacion_k)(combinacion, k)
    for exactitud, combinacion in mejores_5  #lo que obtuve en la funcion anterior
    for k in valores_k
)

# Ordeno resultados por exactitud en orden descendente
resultados_ordenados_k = sorted(resultados_k, key=lambda x: x[0], reverse=True)

# Imprimo los resultados para cada valor de k
resultados_por_k = {k: [] for k in valores_k}
for exactitud, combinacion, k in resultados_ordenados_k:
    resultados_por_k[k].append((exactitud, combinacion))

for k in valores_k:
    print(f"\nResultados para k={k}:")
    for i, (exactitud, combinacion) in enumerate(resultados_por_k[k], 1):
        print(f"Combinación #{i}: {combinacion}")
        print(f"Exactitud: {exactitud}")
        
############
#Como la funcion elige atributos aleatorios, pero en la última prueba el mejor resultado de 10 atributos me dio:
# Con  k=9:
# Combinación #1: [649, 224, 383, 575, 602, 490, 596, 548, 564, 170]
# Exactitud:  0.9739583333333334
#%% ===========================================================================
# 3. (Clasificación multiclase) Dada una imagen se desea responder la
# siguiente pregunta: ¿A cuál de las vocales corresponde la imagen?
#==============================================================================

#--------------------------------------------------------
# a. Vamos a trabajar con los datos correspondientes a las 5 vocales.
# Primero filtrar solo los datos correspondientes a esas letras. Luego,
# separar el conjunto de datos en desarrollo (dev) y validación (held-out).
# Para los incisos b y c, utilizar el conjunto de datos de desarrollo. Dejar
# apartado el conjunto held-out en estos incisos.
#--------------------------------------------------------

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

#%%------------------------------------------------------
# b. Ajustar un modelo de árbol de decisión. Probar con distintas
# profundidades.
#--------------------------------------------------------

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

#%%------------------------------------------------------
# c. Realizar un experimento para comparar y seleccionar distintos árboles
# de decisión, con distintos hiperparámetos. Para esto, utilizar validación
# cruzada con k-folding. ¿Cuál fue el mejor modelo? Documentar cuál
# configuración de hiperparámetros es la mejor, y qué performance
# tiene.
#--------------------------------------------------------

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

#%%------------------------------------------------------
# d. Entrenar el modelo elegido a partir del inciso previo, ahora en todo el
# conjunto de desarrollo. Utilizarlo para predecir las clases en el conjunto
# held-out y reportar la performance.
# OBS: Al realizar la evaluación utilizar métricas de clasificación
# multiclase como por ejemplo la exactitud. Además pueden realizar una
# matriz de confusión y evaluar los distintos tipos de errores para las
# clases.
#--------------------------------------------------------

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
