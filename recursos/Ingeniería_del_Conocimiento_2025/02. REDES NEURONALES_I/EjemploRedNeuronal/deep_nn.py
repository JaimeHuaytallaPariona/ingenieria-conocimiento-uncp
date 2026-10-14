# -*- coding: utf-8 -*-
import numpy as np


# ====================================================================
def sigmoid(z):
    """
    Argumentos
        z - Arreglo de numpy de cualquier tamaño
    Retorna
        Salida de la función sigmoidea
    """
    A = 1/(1+np.exp(-z))
    return A


def relu(z):
    """
    Argumentos
        z - Arreglo de numpy de cualquier tamaño
    Retorna
        Salida de ReLU
    """
    A = np.maximum(0,z)
    return A


# ====================================================================
class DeepNeuralNetwork(object):
    """
    Red Neuronal profunda con L-1 capas ReLU y una capa sigmoidea
    
    """
    def __init__(self, dim_layers, learning_rate=0.01, num_iter=5000, 
                 verbose=False, intervalos=1000):
        """
        Argumentos
            dim_layers - Lista [nx, n1, n2, ..., nL] de tamaño L+1 que contiene el 
                tamaño de cada una de las L capas, incluyendo el tamaño nx de la 
                capa de entrada
            learning_rate - Factor de aprendizaje del descenso de gradiente
            num_iter - Número de iteraciones del descenso de gradiente
            verbose - Muestra el costo cada "intervalos" iteraciones (si es True)
            intervalos - Cada cuántos intervalos se muestra si verbose es True
        
        """
        self.dim_layers = dim_layers
        self.num_layers = len(dim_layers)-1
        self.alfa = learning_rate
        self.num_iter = num_iter
        self.verbose = verbose
        self.intervalos = intervalos
        self.costo = np.zeros(num_iter)
        # Inicialización aleatoria de parámetros
        np.random.seed(1)
        self.params = {}
        for l in range(1, len(self.dim_layers)):
            f = 1 / np.sqrt(self.dim_layers[l-1])
            self.params['W'+str(l)] = f*np.random.randn(self.dim_layers[l], 
                                                        self.dim_layers[l-1]) 
            self.params['b'+str(l)] = np.zeros((self.dim_layers[l], 1))


    def fit(self, X, Y):
        """
        Entrena la red neuronal profunda
        
        Argumentos
            X - Matriz de tamaño (d,n) que contiene n instancias de tamaño d cada una
            Y - Vector de tamaño (1,n) que contiene las etiquetas verdaderas

        """
        # Bucle de optimización de parámetros (descenso de gradiente)
        for i in range(0, self.num_iter):
            # Propagación hacia adelante
            AL, val_interm = self.forward_propagation(X)
            # Propagación hacia atrás (backward propagation)
            gradientes = self.backward_propagation(AL, Y, val_interm)
            # Actualizar parámetros
            for l in range(self.num_layers):
                self.params["W"+str(l+1)] -= self.alfa*gradientes["dW"+str(l+1)]
                self.params["b"+str(l+1)] -= self.alfa*gradientes["db"+str(l+1)]
            # Calcular el costo
            self.costo[i] = self.calcular_costo(AL, Y)
            # Mostrar el costo
            if self.verbose and i%self.intervalos==0:
                costo = np.round(self.costo[i], 4)
                print("Costo luego de la iteración {}: {}".format(i, costo))


    def forward_propagation(self, X):
        """
        Propagación hacia adelante usando L-1 funciones de activación ReLU y 1 
        función sigmoidea para la capa L
        
        Argumentos
            X - Matriz de tamaño (nx, n) de instancias de entrada
        
        Retorna
            AL - último valor de activación
            val_interm - lista de (Aprev, W, b), Z de cada capa oculta (L-1) y 
                         de cada capa de salida (sigmoidea)
    
        """
        val_interm = []        # Almacenamiento de Aprev, Ws, Zs
        L = self.num_layers
        A = X                  # Inicialización de A0 con los datos de entrada
            
        # Función de activación ReLU para las capas ocultas
        for l in range(1, self.num_layers):
            A_prev = A 
            W = self.params['W'+str(l)]; b = self.params['b'+str(l)]
            Z = np.dot(W,A_prev) + b
            A = relu(Z)
            val_interm.append( (A_prev, W, Z)  )    # Valores intermedios

        # Función de activación sigmoide para la última capa
        A_prev = A
        W = self.params['W'+str(L)]; b = self.params['b'+str(L)]
        Z = np.dot(W,A_prev) + b
        AL = sigmoid(Z)
        val_interm.append( (A_prev, W, Z)  )     # Valores intermedios
              
        return AL, val_interm
    
    
    def backward_propagation(self, AL, Y, interm):
        """
        Propagación hacia atrás considerando que la última capa es sigmomidea y 
        todas las demás son ReLU
        
        Argumentos
            AL - Vector de probabilidad que es la salida de la propagación hacia adelante
            Y - Vector que contiene las etiquetas verdaderas
            interm - Lista de (Aprev, W, b), Z de cada capa (ocultas y de salida)
        
        Retorna
            Diccionario con los gradientes dAi, dWi, dbi
    
        """
        grads = {}
        L = self.num_layers
        Y = Y.reshape(AL.shape) # Y tendrá el mismo tamaño de AL
        
        # Inicializáción: derivada con respecto a la activación de la capa sigmoidea (capa L)
        dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))

        # Gradientes para la capa L (capa sigmoidea)
        A_prev, W, Z = interm[L-1]   #interm_actual
        s = 1/(1+np.exp(-Z))
        dZ = dAL*s*(1-s)
        n = A_prev.shape[1]
        grads["dW"+str(L)] = 1.0/n * np.dot(dZ, A_prev.T)
        grads["db"+str(L)] = 1.0/n * np.sum(dZ, axis=1, keepdims=True)
        grads["dA"+str(L-1)] = np.dot(W.T, dZ)
                
        for l in reversed(range(L-1)):
            # Gradientes de cada capa ReLU
            A_prev, W, Z = interm[l]
            dZ = np.array(grads["dA"+str(l+1)], copy=True)  # Conversión de dZ al objeto correcto
            dZ[Z <= 0] = 0                                  # cuando z <= 0, dZ también debe ser 0
            n = A_prev.shape[1]
            grads["dW"+str(l+1)] = 1.0/n*np.dot(dZ, A_prev.T)
            grads["db"+str(l+1)] = 1.0/n*np.sum(dZ, axis=1, keepdims=True)
            grads["dA"+str(l)] = np.dot(W.T, dZ)
        
        return grads
    
   
    def calcular_costo(self, AL, Y):
        """
        Argumentos
            AL - vector (1,n) de probabilidad que corresponde a la predicción
            Y -- Vector (1,n) de verdaderas etiquetas
        Retorna
            costo
        """
        n = Y.shape[1]
        # loss usando AL, Y
        costo = (1.0/n) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        return np.squeeze(costo)

    
    # ====================================================================
    def predict(self, X, umbral=0.5):
        """
        Predecir una clase para cada instancia usando los parámetros entrenados
        de una red neuronal de L capas
        
        Argumentos
            X - Matriz de tamaño (nx, n) que contiene n instancias
    
        Retorna:
            Predicciones para X
        """
        AL, tmp = self.forward_propagation(X)
        # Clasificar usando un umbral 
        predicciones = (AL > umbral)
        return predicciones
    
