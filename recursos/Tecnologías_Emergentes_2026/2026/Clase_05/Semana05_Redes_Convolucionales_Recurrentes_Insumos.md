# Insumo de Investigación — Semana 5
## Redes Convolucionales y Recurrentes Aplicadas
### Tecnologías Emergentes (ISO46B) — FIS-UNCP — Unidad II

*Documento preparado como insumo de investigación — no constituye la presentación final. Uso previsto: base para que el docente diseñe diapositivas, guion de clase y actividades prácticas de la Semana 5.*

---

## Contexto y continuidad con la Semana 4

En la Semana 4 los estudiantes construyeron su primer perceptrón multicapa (MLP) con Keras, trabajando sobre datos tabulares (variables numéricas: temperatura, amplitud térmica). Esta semana da el siguiente paso natural: introducir dos familias de arquitecturas especializadas —**redes convolucionales (CNN)**, diseñadas para datos con estructura espacial como imágenes, y **redes recurrentes (RNN/LSTM)**, diseñadas para datos con estructura secuencial/temporal— y cerrar con buenas prácticas de entrenamiento y ajuste de hiperparámetros que aplican a cualquier red neuronal, incluyendo el MLP de la semana anterior.

Por continuidad y coherencia regional, el bloque de RNN/LSTM reutiliza el dataset climático de Huancayo (`pronostico_huancayo.csv`) ya construido en semanas previas, esta vez tratado como **serie temporal** en lugar de tabla de filas independientes.

---

## 1. Redes neuronales convolucionales (CNN) para clasificación de imágenes

### 1.1 Por qué una red densa (MLP) no es ideal para imágenes

Un MLP trata cada píxel de una imagen como una característica independiente, "aplanando" la imagen en un vector largo antes de la primera capa. Esto tiene dos problemas:

1. **Explosión de parámetros:** una imagen pequeña de 64×64 píxeles en color ya tiene 12,288 valores de entrada; conectar esa entrada a una capa oculta de tamaño razonable produce millones de parámetros, la mayoría innecesarios.
2. **Pérdida de estructura espacial:** al aplanar la imagen, el MLP pierde la información de que un píxel está "al lado" de otro. Dos imágenes con el mismo objeto desplazado unos píxeles se ven completamente distintas para un MLP.

Las CNN resuelven ambos problemas usando **filtros (kernels)** que se deslizan sobre la imagen completa, compartiendo los mismos pesos en toda la imagen y preservando la relación espacial entre píxeles vecinos.

### 1.2 La operación de convolución

Un filtro (por ejemplo, de 3×3 píxeles) se desliza sobre la imagen de entrada, y en cada posición calcula una suma ponderada de los píxeles bajo el filtro, produciendo un único valor de salida. Al repetir esto en toda la imagen se genera un **mapa de características** (*feature map*): una versión reducida de la imagen que resalta un patrón específico (bordes verticales, esquinas, texturas, etc., según qué haya "aprendido" el filtro durante el entrenamiento).

Conceptos clave a explicar en clase:
- **Filtro/kernel:** la matriz pequeña de pesos que se desliza sobre la imagen. Sus valores se aprenden durante el entrenamiento, igual que los pesos de un MLP.
- **Stride:** cuántos píxeles avanza el filtro en cada paso (stride=1 es el más común en capas introductorias).
- **Padding:** si se agregan píxeles de relleno en los bordes para controlar el tamaño de salida (`"same"` mantiene el tamaño, `"valid"` lo reduce).
- **Múltiples filtros por capa:** una capa convolucional no aplica un solo filtro, sino decenas (ej. 32), cada uno aprendiendo a detectar un patrón distinto; la salida de la capa es una pila de mapas de características, uno por filtro.

### 1.3 Pooling: reducción de dimensionalidad

Tras una o más capas convolucionales, es común aplicar una capa de **max pooling**: se divide el mapa de características en bloques pequeños (ej. 2×2) y se conserva solo el valor máximo de cada bloque. Esto reduce el tamaño espacial de los datos (menos cómputo en capas posteriores) y aporta cierta invariancia a pequeños desplazamientos del objeto en la imagen.

### 1.4 Arquitectura típica de una CNN sencilla

```
Entrada (imagen) → [Conv2D → Pooling] × N → Flatten → Dense → Salida
```

Las capas convolucionales y de pooling extraen características visuales progresivamente más abstractas (bordes → texturas → formas → objetos); al final, `Flatten` aplana el resultado a un vector, y una o más capas `Dense` (igual que en el MLP de la Semana 4) realizan la clasificación final.

### 1.5 Implementación práctica con Keras

Para esta introducción se recomienda usar **Fashion-MNIST**, un dataset incluido directamente en Keras (`keras.datasets.fashion_mnist`): 70,000 imágenes en escala de grises de 28×28 píxeles, de 10 categorías de prendas de vestir (camiseta, pantalón, zapato, etc.). Se prefiere sobre el clásico MNIST de dígitos porque presenta un reto ligeramente mayor (visualmente más variado) sin dejar de ser pequeño, rápido de entrenar en CPU, y sin requerir descarga ni preparación manual.

```python
from tensorflow import keras
from tensorflow.keras import layers

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalizar los valores de píxel a [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Keras espera una dimensión de "canal" explícita: (alto, ancho, canales)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

modelo_cnn = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),  # 10 clases → softmax, no sigmoid
])

modelo_cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # etiquetas como enteros (0-9), no one-hot
    metrics=["accuracy"],
)
```

Nota pedagógica importante: a diferencia del clasificador binario de la Semana 4 (salida `sigmoid` + `binary_crossentropy`), aquí hay **10 clases**, por lo que la capa de salida usa `softmax` (produce una distribución de probabilidad sobre las 10 clases) y la pérdida es `sparse_categorical_crossentropy` (apropiada cuando las etiquetas son enteros simples, como en Fashion-MNIST, en vez de vectores one-hot).

**Recomendación didáctica:** antes de entrenar el modelo completo, mostrar en clase 8-10 imágenes de ejemplo con `matplotlib.pyplot.imshow()` junto a su etiqueta, para que los estudiantes vean concretamente qué está clasificando la red. Si el tiempo lo permite, también es muy efectivo visualizar los mapas de características de la primera capa convolucional tras entrenar (extrayendo la salida de esa capa con un modelo auxiliar de Keras), para que los estudiantes vean con sus propios ojos qué patrones (bordes, contornos) detecta cada filtro.

---

## 2. Redes recurrentes (RNN/LSTM) para datos secuenciales

### 2.1 Por qué se necesita una arquitectura distinta para datos secuenciales

Tanto el MLP como la CNN asumen que las entradas son independientes entre sí (o que su relación es puramente espacial, en el caso de CNN). Pero en datos secuenciales —una serie de temperaturas día a día, una oración palabra por palabra, una serie de sensores en el tiempo— **el orden importa** y el valor en un instante depende de los valores anteriores. Ninguna de las arquitecturas anteriores tiene una noción de "memoria" a través del tiempo.

Una **red neuronal recurrente (RNN)** resuelve esto procesando la secuencia paso a paso, manteniendo un **estado oculto** que se actualiza en cada paso y que resume la información relevante de los pasos anteriores. Ese estado se pasa de un paso de tiempo al siguiente, actuando como una forma de "memoria" de corto plazo.

### 2.2 El problema del gradiente que se desvanece, y por qué LSTM lo resuelve

Una RNN simple (`SimpleRNN` en Keras) tiene dificultad para aprender dependencias de largo plazo: al retropropagar el error a través de muchos pasos de tiempo, el gradiente tiende a volverse extremadamente pequeño (se "desvanece"), por lo que la red efectivamente "olvida" información de muchos pasos atrás.

Las redes **LSTM (Long Short-Term Memory)** resuelven este problema introduciendo una **celda de memoria** explícita y tres "compuertas" (*gates*) que controlan el flujo de información:
- **Compuerta de olvido (forget gate):** decide qué parte de la memoria anterior descartar.
- **Compuerta de entrada (input gate):** decide qué información nueva agregar a la memoria.
- **Compuerta de salida (output gate):** decide qué parte de la memoria usar para producir la salida en ese paso.

Para esta introducción no es necesario que los estudiantes deduzcan las ecuaciones matemáticas de las compuertas; basta con la intuición de que **LSTM tiene un mecanismo explícito para decidir qué recordar y qué olvidar**, a diferencia de una RNN simple que solo tiene un estado oculto que se sobrescribe en cada paso.

### 2.3 Convertir una serie temporal en un problema de aprendizaje supervisado: ventaneo (windowing)

Para entrenar una RNN/LSTM sobre una serie temporal, primero hay que transformarla en pares (entrada, salida) usando una **ventana deslizante**: se usan los últimos *N* valores para predecir el siguiente.

Ejemplo con el dataset climático de Huancayo: usando una ventana de 5 días, la temperatura máxima de los días 1-5 se usa como entrada (`X`) para predecir la temperatura máxima del día 6 (`y`); luego se desliza la ventana un día y se repite.

```python
import numpy as np
import pandas as pd

df = pd.read_csv("pronostico_huancayo.csv").sort_values("fecha")
serie = df["temp_max"].values

VENTANA = 5

X, y = [], []
for i in range(len(serie) - VENTANA):
    X.append(serie[i : i + VENTANA])
    y.append(serie[i + VENTANA])

X = np.array(X)
y = np.array(y)

# Keras espera forma (muestras, pasos_de_tiempo, características)
X = X.reshape(-1, VENTANA, 1)

print("X:", X.shape, " y:", y.shape)
```

### 2.4 Implementación práctica con Keras

```python
from tensorflow import keras
from tensorflow.keras import layers

modelo_lstm = keras.Sequential([
    layers.Input(shape=(VENTANA, 1)),
    layers.LSTM(16, activation="tanh"),
    layers.Dense(1),  # regresión: predice un valor numérico continuo, sin activación
])

modelo_lstm.compile(
    optimizer="adam",
    loss="mse",          # error cuadrático medio: métrica estándar para regresión
    metrics=["mae"],      # error absoluto medio, más interpretable (mismas unidades que la variable)
)
```

Nota pedagógica: este es un problema de **regresión** (predecir un valor numérico continuo — la temperatura de mañana), no de clasificación como en los ejemplos anteriores. Por eso la capa de salida no lleva función de activación, y se usa `mse`/`mae` en vez de `crossentropy`/`accuracy`.

**Recomendación didáctica:** antes de entrenar la LSTM, calcular un **baseline ingenuo** ("la temperatura de mañana será igual a la de hoy": `prediccion = X[:, -1]`) y su error (MAE). Este baseline —igual que la regresión logística de la Semana 4— da un punto de referencia honesto: si la LSTM no logra superar claramente a "predecir que mañana será igual que hoy", eso es información valiosa sobre las limitaciones del dataset (pocas muestras, poca variabilidad), no un fracaso del código.

---

## 3. Entrenamiento con datos propios y ajuste de hiperparámetros

### 3.1 Hiperparámetros clave que ya han aparecido, ahora explicados en conjunto

A lo largo de las Semanas 4 y 5, los estudiantes ya usaron varios hiperparámetros sin necesariamente detenerse a definirlos como grupo. Vale la pena consolidarlos en esta sesión:

| Hiperparámetro | Qué controla | Efecto de aumentarlo demasiado | Efecto de reducirlo demasiado |
|---|---|---|---|
| `epochs` | Cuántas veces la red recorre todo el dataset de entrenamiento | Sobreajuste (memoriza en vez de generalizar) | El modelo no llega a aprender (subajuste) |
| `batch_size` | Cuántos ejemplos se procesan antes de actualizar los pesos | Entrenamiento más lento en converger, pero más estable | Entrenamiento más rápido por paso, pero más ruidoso |
| `learning_rate` (dentro del optimizador) | Qué tan grandes son los ajustes a los pesos en cada paso | El entrenamiento puede "saltarse" el mínimo y no converger | El entrenamiento converge muy lentamente |
| N.º de filtros / neuronas por capa | Capacidad del modelo para representar patrones complejos | Sobreajuste, más tiempo de cómputo | El modelo no tiene suficiente capacidad (subajuste) |
| N.º de capas | Profundidad de la red, patrones más abstractos | Sobreajuste, entrenamiento más lento y difícil | Representaciones demasiado simples |

### 3.2 Diagnóstico con curvas de entrenamiento vs. validación (continuidad de la Semana 4)

El mismo patrón de gráficos de `loss`/`val_loss` usado en la Semana 4 permite diagnosticar el comportamiento del modelo:
- **Ambas curvas bajan y se estabilizan juntas:** buen ajuste.
- **La curva de entrenamiento sigue bajando pero la de validación sube:** sobreajuste (*overfitting*) — el modelo está memorizando el conjunto de entrenamiento.
- **Ambas curvas se mantienen altas y no bajan:** subajuste (*underfitting*) — el modelo no tiene suficiente capacidad, o necesita más épocas, o el `learning_rate` es inadecuado.

### 3.3 Early stopping: detener el entrenamiento automáticamente

Una técnica sencilla y muy efectiva para evitar el sobreajuste es el *callback* `EarlyStopping` de Keras, que detiene el entrenamiento automáticamente cuando la pérdida de validación deja de mejorar durante un número de épocas definido (`patience`), y opcionalmente restaura los pesos del mejor punto observado:

```python
from tensorflow.keras.callbacks import EarlyStopping

parada_temprana = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

historial = modelo.fit(
    X_train, y_train,
    epochs=100,  # un número alto: EarlyStopping decidirá cuándo detenerse realmente
    validation_split=0.2,
    callbacks=[parada_temprana],
)
```

### 3.4 Mención breve de dropout (para profundizar en semanas futuras)

Otra técnica común contra el sobreajuste es `layers.Dropout(tasa)`, que "apaga" aleatoriamente una fracción de neuronas en cada paso de entrenamiento, forzando a la red a no depender excesivamente de neuronas individuales. Para esta sesión basta con mencionarla conceptualmente (sin necesariamente implementarla en el laboratorio), como parte del vocabulario que los estudiantes empezarán a encontrar en la literatura y en Unidad II en adelante.

**Recomendación didáctica general para el laboratorio:** proponer un ejercicio guiado de "sobreajuste deliberado" — entrenar un modelo pequeño (CNN o LSTM) durante muchas épocas sin `EarlyStopping` y observar cómo se separan las curvas de entrenamiento y validación; luego repetir con `EarlyStopping` activado y comparar. Ver el problema "ocurrir" y luego "resolverse" en pantalla suele fijar el concepto mejor que la sola definición teórica.

---

## Referencias consultadas

- [Conv2D function of Keras and its use in CNNs — AskPython](https://www.askpython.com/python/examples/conv2d-function)
- [Convolutional Neural Network (CNN): A Complete Guide — LearnOpenCV](https://learnopencv.com/understanding-convolutional-neural-networks-cnn/)
- [Understanding CNNs — convolutions, feature maps, and pooling — InDepth.dev](https://indepth.dev/posts/2007/en/understanding-cnns-convolutions-feature-maps-pooling)
- [A Practical Guide to Convolutional Neural Networks (CNNs) with Keras — Towards Data Science](https://towardsdatascience.com/a-practical-guide-on-convolutional-neural-networks-cnns-with-keras-21421172005e/)
- [Keras Conv2D and Convolutional Layers — PyImageSearch](https://pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/)
- [Understanding Deep Convolutional Neural Networks with Tensorflow and Keras — KDnuggets](https://www.kdnuggets.com/2017/11/understanding-deep-convolutional-neural-networks-tensorflow-keras.html/2)
- [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras — MachineLearningMastery](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
- [Series Forecasting with Recurrent Neural Networks (LSTM) — Cybertec](https://www.cybertec-postgresql.com/en/series-forecasting-with-recurrent-neural-networks-lstm/)
- [Time Series Forecasting with LSTMs using TensorFlow 2 and Keras — Curiousily](https://curiousily.com/posts/time-series-forecasting-with-lstms-using-tensorflow-2-and-keras-in-python/)
- [Time Series Analysis Recurrence Neural Network — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/time-series-analysis-recurrence-neural-network-in-python/)
- [Keras API docs — Conv2D layer](https://keras.io/api/layers/convolutional_layers/convolution2d/)
- [Keras API docs — LSTM layer](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Keras API docs — EarlyStopping callback](https://keras.io/api/callbacks/early_stopping/)
- [Keras datasets — Fashion-MNIST](https://keras.io/api/datasets/fashion_mnist/)

### Bibliografía del curso conectada
- Torres, J. (2020). *Python Deep Learning* — capítulos sobre redes convolucionales y redes recurrentes, coherentes con el enfoque práctico de esta sesión.
- Nagy, Z. (2018). *Artificial Intelligence and Machine Learning Fundamentals* — como marco conceptual general de aprendizaje profundo y su relación con el resto del curso.

---

*Documento preparado como insumo de investigación — no constituye la presentación final. Uso previsto: base para que el docente diseñe diapositivas, guion de clase y actividades prácticas de la Semana 5.*
