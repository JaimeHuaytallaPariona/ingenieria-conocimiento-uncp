# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import h5py


def generar_datos_petalos():
    np.random.seed(1)
    n = 400          # número de instancias
    N = int(n/2)     # número de puntos en cada clase
    X = np.zeros((n,2))                # Matriz de instancias (cada fila es una instancia)
    Y = np.zeros((n,1), dtype='uint8') # Vector de etiquetas (clase 0 y clase 1)
    a = 4                              # Mitad del número de pétalos

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radio
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    return X.T, Y.T


def plot_decision_boundary(modelo, X, y, alpha=0.2):
    # Valores min y max con un delta
    x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
    y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
    h = 0.01
    # Celdas espaciadas una distancia h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predicción según el modelo para todas las celdas
    Xbound = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(Xbound.T)  #(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Gráfico del contorno y de las instancias
    plt.contourf(xx, yy, Z, cmap='winter', alpha=alpha)
    plt.ylabel('$x_2$'); plt.xlabel('$x_1$')
    plt.scatter(X[0, :], X[1, :], c=y.reshape((y.shape[1],)), cmap='winter')


def cargar_imagenes():
    # Cargar imágenes del conjunto de entrenamiento
    train_dataset = h5py.File('datos/train_gatos_nogatos.h5', "r")
    X_train = np.array(train_dataset["train_set_x"][:]) # Atributos
    Y_train = np.array(train_dataset["train_set_y"][:]) # Etiquetas
    # Cargar imágenes del conjunto de prueba (test)
    test_dataset = h5py.File('datos/test_gatos_nogatos.h5', "r")
    X_test = np.array(test_dataset["test_set_x"][:]) # Atributos
    Y_test = np.array(test_dataset["test_set_y"][:]) # Etiquetas
    # Convertir salidas unidimensionales a matrices
    Y_train = Y_train.reshape((1, Y_train.shape[0]))
    Y_test = Y_test.reshape((1, Y_test.shape[0]))
    
    return X_train, Y_train, X_test, Y_test