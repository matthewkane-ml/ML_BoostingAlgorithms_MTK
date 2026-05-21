# Clasificador XGBoost — Predicción de Diabetes

> Gradient boosting aplicado al dataset Pima Indians Diabetes usando el mismo pipeline de EDA contrastado que los proyectos de Árbol de Decisión y Random Forest — un `XGBClassifier(n_estimators=200, learning_rate=0.001)` construido sobre una estrategia de aprendizaje lento que combina muchas correcciones pequeñas en un ensemble predictivo robusto.

---

## Problema

Predecir si un paciente tiene diabetes basándose en medidas diagnósticas. El mismo contexto clínico que los proyectos de Árbol de Decisión y Random Forest — este es el tercer modelo en una progresión desde un árbol individual interpretable, a un bosque con bagging, hasta un ensemble con boosting. La pregunta central: ¿añade la corrección secuencial de errores del gradient boosting un valor medible sobre el bagging?

## Dataset

- **Fuente:** Dataset Pima Indians Diabetes (768 filas × 9 características)
- **Target:** `Outcome` — 1 = diabetes, 0 = no diabetes (distribución de clases 65% / 35%)
- **Características usadas tras limpieza:** Glucose, BMI, Age, Pregnancies (top 4 por SelectKBest f_classif)

El pipeline completo de EDA y preprocesamiento es idéntico al de los proyectos de Árbol de Decisión y Random Forest:

| Paso | Acción |
|---|---|
| Eliminar columnas imposibles | Insulin (48,7% ceros) y SkinThickness (29,6% ceros) eliminadas |
| Imputación de ceros | Imputación con mediana estratificada por grupo en Glucose, BloodPressure, BMI |
| Capping de outliers | Método IQR en Pregnancies, DiabetesPedigreeFunction, Age |
| Escalado | StandardScaler |
| Selección de características | SelectKBest (f_classif, k=4) → Glucose, BMI, Age, Pregnancies |
| División | 80/20 estratificada (614 entrenamiento / 154 prueba) |

## Modelo

**XGBClassifier(n_estimators=200, learning_rate=0.001, random_state=42)**

XGBoost construye árboles secuencialmente — cada árbol ajusta los errores residuales de todos los árboles anteriores. Esto es fundamentalmente diferente de un Random Forest, donde los árboles son independientes y votan en paralelo. Con boosting, los árboles posteriores se dirigen específicamente a los casos que el modelo actualmente falla.

Las elecciones de hiperparámetros reflejan una **estrategia de aprendizaje lento** deliberada:
- `n_estimators=200` — 200 árboles secuenciales, suficiente profundidad para acumular muchas correcciones pequeñas
- `learning_rate=0.001` — cada árbol contribuye solo un pequeño paso hacia la predicción correcta, evitando que cualquier árbol individual sobreajuste los residuos

Esta combinación — muchos árboles, cada uno contribuyendo muy poco — es un enfoque de regularización conocido en gradient boosting. La contrapartida: mayor tiempo de entrenamiento a cambio de una generalización más suave.

## La Distinción Boosting vs. Bagging

| Enfoque | Árboles | Cómo se combinan | Objetivo |
|---|---|---|---|
| Random Forest (bagging) | En paralelo, independientes | Voto mayoritario | Reducir varianza |
| XGBoost (boosting) | Secuenciales, dependientes | Ajuste residual aditivo | Reducir sesgo |

Un Random Forest construye 60 árboles diversos simultáneamente y los promedia para cancelar el ruido. XGBoost construye 200 árboles uno por uno, cada uno corrigiendo lo que el modelo anterior falló — reduciendo progresivamente el error de entrenamiento.

## Conclusiones Clave

- **El boosting corrige el sesgo; el bagging reduce la varianza:** Un Random Forest promedia árboles independientes para reducir la varianza de predicción. XGBoost en cambio se dirige a las muestras que el modelo falla consistentemente, reduciendo iterativamente el sesgo. Estas son estrategias complementarias para diferentes modos de fallo.
- **La tasa de aprendizaje y n_estimators son hiperparámetros acoplados:** Un `learning_rate` muy bajo (0,001) solo es útil si `n_estimators` es suficientemente alto para compensar — cada árbol hace una contribución tan pequeña que se necesitan cientos para acumular una predicción útil. Ajustar uno sin el otro produce un modelo subóptimo.
- **XGBoost funciona con las mismas características preprocesadas que los modelos más simples:** A diferencia de las redes neuronales, el gradient boosting sobre datos tabulares generalmente funciona bien con el mismo pipeline de ingeniería de características usado para Árboles de Decisión y Regresión Logística — no se requieren cambios arquitectónicos.

## Stack Tecnológico

`Python` · `xgboost` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Ejecutar Localmente

```bash
git clone https://github.com/matthewkane-ml/ML_BoostingAlgorithms_MTK.git
cd ML_BoostingAlgorithms_MTK
pip install -r requirements.txt
jupyter notebook src/BoostRevised.ipynb
```

El modelo entrenado se guarda en `models/` mediante `pickle`.

## Próximos Pasos

- Ejecutar `GridSearchCV` sobre `n_estimators`, `learning_rate`, `max_depth` y `subsample` para encontrar la combinación óptima — la configuración de aprendizaje lento actual es fundamentada pero no está ajustada
- Comparar directamente con el Random Forest (n_estimators=60, sin ajuste) y el Árbol de Decisión optimizado (max_depth=5) sobre la misma división de prueba para medir la ganancia real de precisión del boosting
- Añadir **valores SHAP** para explicar predicciones individuales — XGBoost se integra de forma nativa con la librería SHAP, permitiendo atribución de características por paciente sin sacrificar la potencia del modelo

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
