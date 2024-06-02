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
import matplotlib as mlp
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, 
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score)

import random
from joblib import Parallel, delayed
from plot_letters import flip_rotate

data = pd.read_csv("emnist_letters_tp.csv", header=None)
mlp.rcParams['figure.dpi'] = 200

#%%============================================================================
# 1. Realizar un análisis exploratorio de los datos. Entre otras cosas, deben
# analizar la cantidad de datos, cantidad y tipos de atributos, cantidad de clases
# de la variable de interés (letras) y otras características que consideren
# relevantes. Además se espera que con su análisis puedan responder las
# siguientes preguntas:
#==============================================================================

min_max_data = data.copy()
min_max_data[min_max_data.columns[1:]] = MinMaxScaler().fit_transform(min_max_data[min_max_data.columns[1:]])

cant_filas = min_max_data.count(axis=1)
cant_cols = min_max_data.count(axis=0)
clases_letras = min_max_data[0].unique()

#%%------------------------------------------------------
# a. ¿Cuáles parecen ser atributos relevantes para predecir la letra a la que
# corresponde la imagen? ¿Cuáles no? ¿Creen que se pueden
# descartar atributos?
#--------------------------------------------------------

media_por_clase = min_max_data.groupby(0).mean()

fig, ax = plt.subplots(6,5, figsize=(10,10))

it = media_por_clase.iterrows()

for i in range(6):
    for j in range(5):
        try:
            let, vals = next(it)
            
            sns.heatmap(flip_rotate(vals.to_numpy()), ax=ax[i,j], vmin=0, vmax=1, cmap='mako')
            ax[i,j].set_title(let)
            ax[i,j].axis('off')
        except:
            ax[i,j].remove()

plt.tight_layout(w_pad=0.8, h_pad=1.06)
plt.show()

#%%

fig, ax = plt.subplots()

std_entre_clase = media_por_clase.std()

sns.heatmap(flip_rotate(std_entre_clase.to_numpy()), cmap='mako',ax=ax)
ax.axis('off')
plt.show()

#%%------------------------------------------------------
# b. ¿Hay letras que son parecidas entre sí? Por ejemplo, ¿Qué es más
# fácil de diferenciar: las imágenes correspondientes a la letra E de las
# correspondientes a la L, o la letra E de la M?
#--------------------------------------------------------

esp_E = media_por_clase.loc['E']
esp_L = media_por_clase.loc['L']
esp_M = media_por_clase.loc['M']

print(f"Coeficiente de correlación para E y L: {esp_E.corr(esp_L)}")
print(f"Coeficiente de correlación para E y M: {esp_E.corr(esp_M)}")

trans_mpc = media_por_clase.transpose()
corr_matrix = trans_mpc.corr()

fig, ax = plt.subplots(figsize=(8,7))

sns.heatmap(corr_matrix, ax=ax, cmap=sns.color_palette("mako_r", as_cmap=True))
ax.xaxis.tick_top()
ax.tick_params(axis='y', labelrotation=0)
ax.set_xlabel("")
ax.set_ylabel("")

plt.show()

#%%------------------------------------------------------
# c. Tomen una de las clases, por ejemplo la letra C, ¿Son todas las
# imágenes muy similares entre sí?
#--------------------------------------------------------

std_C = min_max_data[min_max_data[0] == 'C'].drop(0, axis=1).std()

fig, ax = plt.subplots()

sns.heatmap(flip_rotate(std_C.to_numpy()), cmap='mako',ax=ax)
ax.axis('off')

print(f'Media de la desviación estandar para la letra "C": {std_C.mean()}')

plt.show()

#%% Media de la std para todas las letras

fig, ax = plt.subplots()

std_por_clase = min_max_data.groupby(0).std()
media_std = std_por_clase.mean(axis=1)

sns.barplot(x=media_std.index.to_numpy(), y=media_std.to_numpy(), ax=ax)
ax.set_ylabel("Media de la desviación estandar")
plt.show()

#%%============================================================================
# 2.(Clasificación binaria) Dada una imagen se desea responder la siguiente
# pregunta: ¿la imagen corresponde a la letra L o a la letra A?
#==============================================================================

data[data.columns[1:]] = StandardScaler().fit_transform(data[data.columns[1:]])

#%%------------------------------------------------------
# a. A partir del dataframe original, construir un nuevo dataframe que
# contenga sólo al subconjunto de imágenes correspondientes a las
# letras L o A.
#--------------------------------------------------------

letras_A_L = data[(data[0] == 'A') | 
                  (data[0] == 'L')]

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1,
                                                    stratify=Y, 
                                                    test_size = 0.2)

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

num_muestras = 100
knn = KNeighborsClassifier(n_neighbors=3)
atributos = X_train.columns.tolist()

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
# Como la funcion elige atributos aleatorios, pero en la última prueba el mejor 
# resultado de 10 atributos me dio:
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

alturas = range(1,22,2) 

resultados_3b = np.zeros(len(alturas))

for i, hmax in enumerate(alturas):
    modelo = tree.DecisionTreeClassifier(max_depth = hmax)
    modelo.fit(X_train, Y_train)

    resultados_3b[i] = modelo.score(X_train, Y_train)
    print(f"Profundidad {hmax}: Performance {resultados_3b[i]}")
    
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
alturas = range(7,15)
nsplits = 10

kf = KFold(n_splits=nsplits, 
           shuffle=True, 
           random_state=1)

test_gini = np.zeros((nsplits, len(alturas)))
train_gini = np.zeros((nsplits, len(alturas)))

test_entropy = np.zeros((nsplits, len(alturas)))
train_entropy = np.zeros((nsplits, len(alturas)))

for i, (train_index, test_index) in enumerate(kf.split(X_train)):

    kf_X_train, kf_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
    kf_Y_train, kf_Y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]

    for j, h in enumerate(alturas):
        # Criterio Gini
        modelo = tree.DecisionTreeClassifier(criterion='gini',
                                             max_depth=h, 
                                             random_state=1)
        modelo.fit(kf_X_train, kf_Y_train)
        test_gini[i,j] = modelo.score(kf_X_test, kf_Y_test)
        train_gini[i,j] = modelo.score(kf_X_train, kf_Y_train)
        
        # Criterio Entropy
        modelo = tree.DecisionTreeClassifier(criterion='entropy',
                                             max_depth=h,
                                             random_state=1)
        modelo.fit(kf_X_train, kf_Y_train)
        test_entropy[i,j] = modelo.score(kf_X_test, kf_Y_test)
        train_entropy[i,j] = modelo.score(kf_X_train, kf_Y_train)      

#%%          
scores_promedio_test_gini = test_gini.mean(axis = 0)
scores_promedio_train_gini = train_gini.mean(axis = 0)
scores_promedio_test_entropy = test_entropy.mean(axis = 0)
scores_promedio_train_entropy = train_entropy.mean(axis = 0)

scores_promedio_test_gini_df = pd.DataFrame(scores_promedio_test_gini, index=alturas, columns=['Test_Gini'])
scores_promedio_train_gini_df = pd.DataFrame(scores_promedio_train_gini, index=alturas, columns=['Train_Gini'])
scores_promedio_test_entropy_df = pd.DataFrame(scores_promedio_test_entropy, index=alturas, columns=['Test_Entropy'])
scores_promedio_train_entropy_df = pd.DataFrame(scores_promedio_train_entropy, index=alturas, columns=['Train_Entropy'])

test_scores = pd.concat([scores_promedio_test_gini_df, scores_promedio_test_entropy_df], axis=1)
train_scores = pd.concat([scores_promedio_train_gini_df, scores_promedio_train_entropy_df], axis=1)

print(test_scores)
print("\n",train_scores)

#%% Performance vs profundidad del modelo de árbol de decisión
plt.figure(figsize=(12, 6))

plt.plot(alturas, scores_promedio_test_gini, label='Test-gini', marker='o')
plt.plot(alturas, scores_promedio_test_entropy, label='Test-entropy', marker='v')

plt.plot(alturas, scores_promedio_train_gini, label='Train-gini', marker='o')
plt.plot(alturas, scores_promedio_train_entropy, label='Train-entropy', marker='v')

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

# Mejor modelo: max_depth = 10, criterion = entropy
modelo = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)
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

print(f"Accuracy {accuracy_3d} \nPrecision {precision_3d} \nRecall {recall_3d} \nF1 {f1_3d}")