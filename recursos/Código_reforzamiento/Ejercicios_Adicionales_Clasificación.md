# Ejercicios Propuestos — Modelos de Clasificación
## Material de Refuerzo — Ingeniería del Conocimiento (ISO56B)
### Msc. Jaime Antonio Huaytalla Pariona — UNCP, 2026-II

---

## Nivel 1: Básico (Comprensión de conceptos)

### Ejercicio 1.1 — Exploración del dataset Wine
**Dataset:** `sklearn.datasets.load_wine()` (178 muestras, 13 features, 3 clases de vino)

**Tareas:**
1. Cargar el dataset y convertirlo a DataFrame
2. Mostrar las dimensiones, nombres de features y distribución de clases
3. Calcular estadísticas descriptivas por clase
4. Crear un pairplot con las 4 features más relevantes (alcohol, malic_acid, flavanoids, color_intensity)
5. Responder: ¿Las 3 clases se separan visualmente? ¿Qué features parecen más discriminantes?

**Entregable:** Notebook con las visualizaciones y respuestas escritas.

---

### Ejercicio 1.2 — Train/Test Split y su efecto
**Dataset:** `sklearn.datasets.load_iris()`

**Tareas:**
1. Entrenar una Regresión Logística con 3 splits diferentes: test_size=0.1, 0.3, 0.5
2. Reportar accuracy en train y test para cada split
3. Repetir el experimento con random_state=0, 7, 42, 99 (fijando test_size=0.2) y reportar cómo varía la accuracy
4. Responder: ¿Por qué cambian los resultados con distinto random_state? ¿Qué solución propones?

**Entregable:** Tabla comparativa y respuesta razonada.

---

### Ejercicio 1.3 — Efecto de la estandarización
**Dataset:** `sklearn.datasets.load_breast_cancer()`

**Tareas:**
1. Entrenar un KNN (K=5) **sin** estandarizar y reportar accuracy
2. Entrenar el mismo KNN **con** StandardScaler y reportar accuracy
3. Entrenar un Árbol de Decisión con y sin estandarización
4. Crear una tabla comparativa con los 4 resultados
5. Responder: ¿Cuál modelo es afectado por la estandarización y cuál no? ¿Por qué?

**Entregable:** Tabla y explicación.

---

## Nivel 2: Intermedio (Aplicación de algoritmos)

### Ejercicio 2.1 — Clasificación de dígitos escritos a mano
**Dataset:** `sklearn.datasets.load_digits()` (1797 imágenes de 8×8 píxeles, 10 clases)

**Tareas:**
1. Cargar el dataset y visualizar 20 imágenes aleatorias con sus etiquetas
2. Dividir en train/test (80/20, estratificado)
3. Entrenar 3 modelos: Regresión Logística, SVM (RBF), KNN (K=5)
4. Para cada modelo: mostrar accuracy, classification_report y matriz de confusión
5. Identificar: ¿Qué dígitos se confunden más entre sí? ¿Tiene sentido visualmente?
6. Crear un gráfico de barras comparando accuracy por modelo

**Entregable:** Notebook completo con análisis.

---

### Ejercicio 2.2 — Búsqueda del K óptimo en KNN
**Dataset:** `sklearn.datasets.load_wine()`

**Tareas:**
1. Estandarizar los datos
2. Probar KNN con K de 1 a 30
3. Para cada K, calcular accuracy en train y test
4. Graficar ambas curvas en un solo gráfico (X: K, Y: accuracy)
5. Identificar la zona de overfitting (K bajo) y underfitting (K alto)
6. Reportar el K óptimo y entrenar el modelo final con ese K
7. Mostrar classification_report del modelo final

**Entregable:** Gráfico anotado y modelo final evaluado.

---

### Ejercicio 2.3 — Comparación de kernels en SVM
**Dataset:** Crear datos sintéticos con `make_moons` y `make_circles`

**Tareas:**
1. Generar 500 muestras con `make_moons(noise=0.3)` y 500 con `make_circles(noise=0.2, factor=0.5)`
2. Para cada dataset, entrenar SVM con kernel lineal, RBF y polinomial (grado 3)
3. Visualizar las fronteras de decisión de cada kernel (gráfico 2D con regiones coloreadas)
4. Reportar accuracy para cada combinación (dataset × kernel)
5. Responder: ¿Por qué el kernel lineal falla en `make_moons`? ¿Qué kernel funciona mejor para cada forma de datos?

**Pista para visualizar fronteras:**
```python
from sklearn.inspection import DecisionBoundaryDisplay
```

**Entregable:** 6 gráficos (2 datasets × 3 kernels) y tabla de accuracies.

---

### Ejercicio 2.4 — Árbol de Decisión: efecto de la profundidad
**Dataset:** `sklearn.datasets.load_breast_cancer()`

**Tareas:**
1. Entrenar árboles con max_depth de 1 a 15
2. Registrar accuracy en train y test para cada profundidad
3. Graficar ambas curvas e identificar el punto de inflexión (overfitting)
4. Visualizar el árbol con profundidad óptima usando `plot_tree`
5. Mostrar la importancia de features como gráfico de barras horizontal
6. Responder: ¿Cuáles son las 5 features más importantes para distinguir tumores malignos de benignos?

**Entregable:** Gráficos y análisis escrito.

---

## Nivel 3: Avanzado (Pipelines, CV, tuning)

### Ejercicio 3.1 — Pipeline completo con Cross-Validation
**Dataset:** Wine Quality Red de UCI (descargar desde sklearn o seaborn)

**Tareas:**
1. Cargar el dataset y binarizar el target: calidad ≥ 7 = "bueno", calidad < 7 = "regular"
2. Verificar el desbalance de clases y comentar
3. Crear pipelines con StandardScaler + cada uno de los 4 modelos (LogReg, Árbol, SVM, KNN)
4. Evaluar cada pipeline con cross_val_score (5-fold, scoring='f1')
5. Crear un boxplot comparativo de los F1-scores por fold
6. Seleccionar el mejor modelo y entrenar en todo el train set
7. Evaluar en test: classification_report, matriz de confusión y accuracy

**Entregable:** Notebook con pipeline, CV y selección documentada.

---

### Ejercicio 3.2 — GridSearchCV para optimizar SVM
**Dataset:** `sklearn.datasets.load_breast_cancer()`

**Tareas:**
1. Crear un pipeline: StandardScaler → SVC
2. Definir un grid de hiperparámetros:
   - C: [0.01, 0.1, 1, 10, 100]
   - kernel: ['linear', 'rbf']
   - gamma: ['scale', 0.001, 0.01, 0.1, 1] (solo para RBF)
3. Ejecutar GridSearchCV con cv=5 y scoring='f1'
4. Reportar los mejores hiperparámetros y el mejor F1-score (CV)
5. Evaluar el mejor modelo en test
6. Crear un heatmap de F1-scores para las combinaciones de C y gamma (kernel RBF)

**Entregable:** Notebook con grid search, heatmap y análisis.

---

### Ejercicio 3.3 — Manejo de clases desbalanceadas
**Dataset:** Crear un dataset desbalanceado con `make_classification`

**Tareas:**
1. Generar 1000 muestras con `make_classification(n_classes=2, weights=[0.95, 0.05])` (95% vs 5%)
2. Entrenar una Regresión Logística SIN manejar el desbalance → reportar accuracy, precision, recall, F1
3. Entrenar con `class_weight='balanced'` → comparar métricas
4. Responder: ¿Por qué la accuracy es alta incluso sin balancear? ¿Qué métrica revela el problema?
5. Graficar las curvas Precision-Recall para ambos modelos
6. Repetir usando SMOTE de `imbalanced-learn` (si está disponible en Colab) y comparar

**Entregable:** Comparación de 3 estrategias con métricas y gráficos.

---

### Ejercicio 3.4 — Heart Disease: Proyecto mini completo
**Dataset:** Heart Disease UCI (cargar desde CSV o usar la versión de sklearn/kaggle)

**Tareas:**
1. **Exploración:** estadísticas, distribuciones, correlaciones, valores faltantes
2. **Preprocesamiento:** manejar valores faltantes (si los hay), estandarizar features numéricas
3. **Modelado:** entrenar al menos 3 modelos dentro de pipelines
4. **Evaluación:** cross-validation (F1), matrices de confusión, curvas ROC
5. **Tuning:** GridSearchCV para el mejor modelo
6. **Selección:** justificar la elección del modelo final
7. **Interpretación:** ¿Qué features son más importantes para predecir enfermedad cardíaca?

**Restricción:** Toda la evaluación debe considerar que un falso negativo (no detectar la enfermedad) es más costoso que un falso positivo.

**Entregable:** Notebook completo tipo proyecto, con narrativa explicando cada decisión.

---

## Nivel 4: Desafío (Integración y pensamiento crítico)

### Ejercicio 4.1 — Comparación justa de modelos con múltiples datasets
**Datasets:** Iris, Wine, Breast Cancer, Digits

**Tareas:**
1. Para cada dataset, entrenar los 4 modelos (LogReg, Árbol, SVM-RBF, KNN) dentro de pipelines
2. Evaluar todos con cross-validation (5-fold, F1-weighted)
3. Crear una tabla de resultados: filas = datasets, columnas = modelos
4. Crear un heatmap de la tabla
5. Responder con argumento:
   - ¿Existe un modelo que sea "el mejor" en todos los datasets?
   - ¿Qué características del dataset favorecen a cada modelo?
   - ¿Qué modelo recomendarías como "primera opción" cuando no se sabe nada del problema?

**Entregable:** Análisis comparativo con tablas, heatmap y argumentación escrita.

---

### Ejercicio 4.2 — Impacto de features irrelevantes
**Dataset:** `sklearn.datasets.load_breast_cancer()`

**Tareas:**
1. Entrenar un SVM (RBF) con las 30 features originales → F1 con CV
2. Añadir 20 features aleatorias (ruido) al dataset → entrenar y reportar F1
3. Añadir 50 features aleatorias → entrenar y reportar F1
4. Graficar F1 vs. número de features ruidosas (0, 10, 20, 30, 40, 50)
5. Repetir el experimento con KNN y Árbol de Decisión
6. Responder: ¿Qué modelo es más robusto al ruido? ¿Por qué?
7. Proponer y aplicar una técnica de selección de features para recuperar el rendimiento

**Entregable:** Gráficos comparativos y análisis de robustez.

---

*Total: 12 ejercicios graduados (3 básicos + 4 intermedios + 3 avanzados + 2 desafíos)*

*Material de refuerzo — Ingeniería del Conocimiento (ISO56B) — UNCP*
