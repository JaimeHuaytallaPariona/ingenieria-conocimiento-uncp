# Ejercicios Propuestos — Modelos de Regresión
## Material de Refuerzo — Ingeniería del Conocimiento (ISO56B)
### Msc. Jaime Antonio Huaytalla Pariona — UNCP, 2026-II

---

## Nivel 1: Básico (Comprensión de conceptos)

### Ejercicio 1.1 — Exploración del dataset Diabetes
**Dataset:** `sklearn.datasets.load_diabetes()` (442 muestras, 10 features, target continuo)

**Tareas:**
1. Cargar el dataset y convertirlo a DataFrame
2. Mostrar dimensiones, nombres de features y estadísticas descriptivas
3. Calcular la correlación de cada feature con el target
4. Crear scatter plots de los 4 features más correlacionados con el target
5. Crear un histograma del target y comentar su distribución
6. Responder: ¿Qué features parecen más útiles para predecir la progresión de la diabetes?

**Entregable:** Notebook con visualizaciones y respuestas escritas.

---

### Ejercicio 1.2 — Mi primera regresión lineal
**Dataset:** `sklearn.datasets.load_diabetes()`

**Tareas:**
1. Seleccionar UN solo feature (bmi) y entrenar una Regresión Lineal Simple
2. Graficar los datos (scatter) con la recta de regresión superpuesta
3. Calcular e interpretar: RMSE, MAE, R²
4. Ahora entrenar con TODOS los features y comparar las métricas
5. Responder: ¿Cuánto mejora el R² al usar todos los features? ¿Por qué tiene sentido?

**Entregable:** Gráfico de la recta, tabla comparativa y explicación.

---

### Ejercicio 1.3 — Efecto del Train/Test Split
**Dataset:** `sklearn.datasets.fetch_california_housing()`

**Tareas:**
1. Entrenar una Regresión Lineal con 4 proporciones de test: 10%, 20%, 30%, 50%
2. Para cada split, reportar R² en train y en test
3. Repetir con random_state = 0, 7, 42, 99 (fijando test_size=0.2) y reportar R²
4. Calcular la desviación estándar de los R² obtenidos con diferentes random_states
5. Responder: ¿Por qué los resultados varían? ¿Qué técnica resolvería este problema?

**Entregable:** Tablas comparativas y respuesta razonada.

---

## Nivel 2: Intermedio (Aplicación de algoritmos)

### Ejercicio 2.1 — Regresión Polinomial: encontrando el grado óptimo
**Dataset:** Datos sintéticos con relación no lineal

**Tareas:**
1. Generar 100 datos con: `y = 3x² - 2x + 5 + ruido`
   ```python
   np.random.seed(42)
   X = np.random.uniform(-3, 3, 100).reshape(-1, 1)
   y = 3*X.ravel()**2 - 2*X.ravel() + 5 + np.random.normal(0, 3, 100)
   ```
2. Dividir en train/test (80/20)
3. Entrenar regresiones polinomiales con grados 1 a 15
4. Graficar Train RMSE y Test RMSE vs. grado del polinomio
5. Identificar el grado óptimo y las zonas de underfitting y overfitting
6. Visualizar las curvas de los modelos con grado 1, grado óptimo y grado 15

**Entregable:** Gráficos anotados y análisis del trade-off sesgo-varianza.

---

### Ejercicio 2.2 — Comparación Ridge vs Lasso
**Dataset:** `sklearn.datasets.fetch_california_housing()`

**Tareas:**
1. Crear un pipeline con StandardScaler + Ridge y otro con StandardScaler + Lasso
2. Probar alpha = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] para ambos modelos
3. Para cada alpha: reportar R² en test y número de coeficientes distintos de cero
4. Graficar R² vs alpha para Ridge y Lasso en el mismo gráfico
5. Graficar la evolución de los coeficientes vs alpha (un gráfico por modelo)
6. Responder: ¿Cuál es la diferencia fundamental entre Ridge y Lasso en cuanto a la selección de features?

**Entregable:** Gráficos comparativos y explicación detallada.

---

### Ejercicio 2.3 — Árbol de Decisión para Regresión: profundidad óptima
**Dataset:** `sklearn.datasets.fetch_california_housing()`

**Tareas:**
1. Entrenar árboles de decisión con max_depth de 1 a 20
2. Registrar R² en train y test para cada profundidad
3. Graficar ambas curvas e identificar el punto de overfitting
4. Visualizar el árbol con profundidad óptima usando `plot_tree` (hasta 3 niveles)
5. Mostrar la importancia de features como gráfico de barras horizontal
6. Comparar: ¿El árbol óptimo supera a la Regresión Lineal en este dataset?

**Entregable:** Gráficos, visualización del árbol y análisis.

---

### Ejercicio 2.4 — Random Forest: número de árboles y profundidad
**Dataset:** `sklearn.datasets.fetch_california_housing()`

**Tareas:**
1. Fijar max_depth=10 y probar n_estimators = [1, 5, 10, 25, 50, 100, 200, 500]
2. Graficar R² test vs número de árboles
3. Fijar n_estimators=100 y probar max_depth = [2, 4, 6, 8, 10, 12, 15, None]
4. Graficar R² test vs max_depth
5. Entrenar el modelo con la combinación óptima encontrada
6. Responder: ¿Más árboles siempre mejora el resultado? ¿Qué pasa con max_depth=None?

**Entregable:** Gráficos y análisis de hiperparámetros.

---

## Nivel 3: Avanzado (Pipelines, CV, tuning)

### Ejercicio 3.1 — Pipeline completo con Cross-Validation
**Dataset:** `sklearn.datasets.fetch_california_housing()`

**Tareas:**
1. Crear pipelines con StandardScaler + cada modelo:
   - Regresión Lineal, Ridge (α=1), Lasso (α=0.01), Elastic Net (α=0.01, l1_ratio=0.5)
   - Árbol de Decisión (max_depth=8), Random Forest (100 árboles), Gradient Boosting
2. Evaluar cada pipeline con `cross_val_score` (5-fold, scoring='neg_root_mean_squared_error')
3. Crear un boxplot comparativo de los RMSE por fold para todos los modelos
4. Seleccionar el mejor modelo y evaluar en test
5. Reportar: RMSE, MAE, R² del modelo ganador en test

**Entregable:** Notebook con comparación completa y selección justificada.

---

### Ejercicio 3.2 — GridSearchCV para Gradient Boosting
**Dataset:** `sklearn.datasets.fetch_california_housing()`

**Tareas:**
1. Crear un Gradient Boosting Regressor
2. Definir un grid de hiperparámetros:
   - n_estimators: [100, 200, 300]
   - learning_rate: [0.01, 0.05, 0.1, 0.2]
   - max_depth: [3, 4, 5, 6]
3. Ejecutar GridSearchCV con cv=5 y scoring='neg_root_mean_squared_error'
4. Reportar los mejores hiperparámetros y el mejor RMSE (CV)
5. Evaluar el mejor modelo en test con todas las métricas (RMSE, MAE, R²)
6. Crear un heatmap de RMSE para las combinaciones de learning_rate y max_depth (fijando el mejor n_estimators)

**Entregable:** Notebook con grid search, heatmap y análisis.

---

### Ejercicio 3.3 — Análisis de residuos completo
**Dataset:** `sklearn.datasets.fetch_california_housing()`

**Tareas:**
1. Entrenar el mejor modelo encontrado en el ejercicio 3.2
2. Generar las predicciones en test
3. Crear 4 gráficos de residuos:
   a. Predicción vs Valor Real (con línea diagonal)
   b. Residuos vs Predicción (buscar patrones)
   c. Histograma de residuos (verificar normalidad)
   d. Q-Q plot de residuos
4. Identificar las 10 predicciones con mayor error absoluto
5. Analizar: ¿Hay algún patrón en los errores más grandes? ¿Qué los causa?
6. Responder: ¿Los residuos sugieren que el modelo es adecuado o hay problemas sistemáticos?

**Entregable:** Los 4 gráficos, análisis de errores grandes y conclusión.

---

## Nivel 4: Desafío (Integración y pensamiento crítico)

### Ejercicio 4.1 — Importancia de features: tres métodos comparados
**Dataset:** `sklearn.datasets.fetch_california_housing()`

**Tareas:**
1. Entrenar tres modelos: Ridge, Random Forest, Gradient Boosting
2. Obtener la importancia de features de 3 formas:
   - Coeficientes Ridge (estandarizados) → |coeficiente|
   - Feature importance de Random Forest (MDI)
   - Permutation Importance (usando `sklearn.inspection.permutation_importance`)
3. Crear un gráfico de barras agrupadas comparando los 3 rankings
4. Responder con argumento:
   - ¿Los 3 métodos coinciden en cuáles son los features más importantes?
   - ¿Cuál método es más confiable y por qué?
   - ¿Podrías eliminar features de baja importancia sin perder rendimiento? Pruébalo.

**Entregable:** Gráfico comparativo, análisis y experimento de eliminación de features.

---

### Ejercicio 4.2 — Predicción con datos reales: Ames Housing
**Dataset:** Ames Housing (descargar desde Kaggle o usar `from sklearn.datasets import fetch_openml; fetch_openml(name='house_prices', as_frame=True)`)

**Tareas:**
1. **Exploración:** Estadísticas, distribuciones, correlaciones, valores faltantes
2. **Preprocesamiento:**
   - Manejar valores faltantes (imputación por mediana o moda según tipo)
   - Codificar variables categóricas (si las hay) con OneHotEncoder
   - Estandarizar features numéricas
3. **Modelado:** Entrenar al menos 4 modelos dentro de pipelines
4. **Evaluación:** Cross-validation (RMSE), gráficos de predicción vs real
5. **Tuning:** GridSearchCV para el mejor modelo
6. **Selección:** Justificar la elección del modelo final con métricas
7. **Interpretación:** ¿Qué features son más importantes para predecir el precio?

**Restricción:** Todo el flujo debe usar pipelines para evitar data leakage.

**Entregable:** Notebook completo tipo proyecto, con narrativa explicando cada decisión.

---

*Total: 12 ejercicios graduados (3 básicos + 4 intermedios + 3 avanzados + 2 desafíos)*

*Material de refuerzo — Ingeniería del Conocimiento (ISO56B) — UNCP*
