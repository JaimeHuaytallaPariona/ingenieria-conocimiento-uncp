# -*- coding: utf-8 -*-
import numpy as np

# ====================================================================
def sigmoid(z):
    """
    Calcula la función sigmoidea de z

    Argumentos
        z - Escalar o arreglo de numpy
    Retorna
        s - sigmoid(z)
    """
    s = 1/(1+np.exp(-z))
    return s


# ====================================================================
class SimpleNeuralNetwork(object):
    """
    Red Neuronal simple de 2 capas: la capa oculta tiene una función de 
    activación tanh, y la capa de salida es sigmoidea.
    """
    
    def __init__(self, nhidden, num_iter=10000, learning_rate=0.1, 
                 verbose=False):
        """
        Argumentos
            nhidden - Número de neuronas en la capa oculta
            num_iter - Número de iteraciones de descenso del gradiente
            learning_rate - factor de aprendizaje para descenso de gradiente
            verbose - Muestra el costo cada 1000 iteraciones (si es True)
        """        
        # Almacenamiento de parámetros
        self.nh = nhidden
        self.num_iter = num_iter
        self.verbose = verbose
        self.alfa = learning_rate
        self.costo = np.zeros(num_iter)

        
    def fit(self, X, y):
        """
        Entrena la red neuronal
        
        Argumentos
             X - Matriz de tamaño (2,n) que contiene n instancias
             Y - Vector de etiquetas de tamaño (1,n)

        """
        nx = X.shape[0]     # Tamaño de la capa de entrada
        ny = y.shape[0]     # Tamaño de la capa de salida
        
        # Inicialización aleatoria de parámetros
        np.random.seed(2)
        W1 = np.random.randn(self.nh, nx)*0.01
        b1 = np.zeros((self.nh, 1))
        W2 = np.random.randn(ny, self.nh)*0.01
        b2 = np.zeros((ny, 1))
        self.params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
        # Bucle de optimización de parámetros (descenso de gradiente)
        for i in range(0, self.num_iter):
            # Propagación hacia adelante
            A2, val_interm = self.forward_propagation(X)
            # Propagación hacia atrás
            gradientes = self.backward_propagation(X, y, val_interm)
            # Actualización de parámetros (descenso de gradiente)
            self.params["W1"] = self.params["W1"] - self.alfa*gradientes["dW1"]
            self.params["b1"] = self.params["b1"] - self.alfa*gradientes["db1"]
            self.params["W2"] = self.params["W2"] - self.alfa*gradientes["dW2"]
            self.params["b2"] = self.params["b2"] - self.alfa*gradientes["db2"]
            # Calcular el costo
            self.costo[i] = self.calcular_costo(A2, y)
            # Mostrar el costo
            if self.verbose and i%1000==0:
                # Función de costo
                costo = np.round(self.costo[i], 4)
                print("Costo luego de la iteración {}: {}".format(i, costo))

    
    def forward_propagation(self, X):
        """
        Argumentos
            X - Matriz de tamaño (nx, n) de instancias de entrada
        
        Retorna
            A2 - Salida sigmoidea de la segunda activación
            val_interm - diccionario que contiene valores intermedios: A1, A2
            
        """
        
        # Recuperar parámetros
        W1 = self.params["W1"]; b1 = self.params["b1"]
        W2 = self.params["W2"]; b2 = self.params["b2"]
        
        # Propagación hacia adelante para calcular la salida A2
        Z1 = W1.dot(X) + b1
        A1 = np.tanh(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = sigmoid(Z2)
        
        # Valores intermedios almacenados temporalmente (caché)
        val_interm = {"A1": A1, "A2": A2}
        return A2, val_interm     


    def backward_propagation(self, X, Y, interm):
        """
        Argumentos
            X - Matriz de tamaño (2,n) de instancias de entrada
            Y - Vector de tamaño (1,n) de etiquetas reales de las instancias
            interm - diccionario que contiene Z1, A1, Z2, A2
        
        Retorna
            diccionario que contiene los gradientes: dW1, db1, dW2, db2
    
        """
        n = X.shape[1]
        
        # Recuperar parámetros
        # W1 = self.params["W1"]
        W2 = self.params["W2"]
        # Recuperar elementos temporales del "cache"
        A1 = interm["A1"]
        A2 = interm["A2"]
        
        # Calcular las derivadas de la función de costo respecto a cada parámetro
        #  Capa 2 (con función de activación sigmoidea)
        dZ2 = A2 - Y
        dW2 = 1/n*np.dot(dZ2, A1.T)
        db2 = 1/n*np.sum(dZ2, axis=1, keepdims=True)
        # Capa 1 (con función de activación tanh)
        dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1,2))
        dW1 = 1/n*np.dot(dZ1, X.T)
        db1 = 1/n*np.sum(dZ1, axis=1, keepdims=True)
        
        gradientes = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return gradientes
    
    
    def calcular_costo(self, A2, y):
        """
        Calcula la función de costo de la red neuronal
        
        Argumentos
            A2 - Vector de predicción (1,n): salida sigmoidea de la segunda activación
             y - Vector de tamaño (1,n) con las etiquetas reales de las instancias
        
        Retorna:
            Costo ("cross-entropy")
        
        """
        # Número de instancias
        n = y.shape[1]
        # Función de costo
        L = y*np.log(A2) + (1-y)*np.log(1-A2)
        costo = -1/n*np.sum(L)
        
        # Retorna un valor escalar puro (no parte de un arreglo)
        return float(np.squeeze(costo))


    def predict(self, X, umbral=0.5):
        """
        Predice una clase para cada instancia usando los parámetros entrenados
        
        Argumentos
            X - Matriz de tamaño (nx, n) que contiene n instancias
            umbral - umbral para la predicción de clase 0 o 1
        
        Retorna
            vector que contiene las predicciones para cada instancia de entrada

        """
        # Salida usando propagación hacia adelante y clasificar como 0 o 1
        A2, tmp = self.forward_propagation(X)
        predicciones = (A2 > umbral)
        return predicciones
    