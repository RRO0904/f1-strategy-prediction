# **Formula 1 Strategy Prediction**
### **Deep Learning aplicado a la predicción de paradas en boxes y cambio de neumáticos**

## **El Problema de Negocio**
En la Fórmula 1, una decisión estratégica tomada en fracciones de segundo puede definir el resultado de una carrera. El objetivo de este proyecto es predecir el momento óptimo y el tipo de neumático para las paradas en boxes (pit stops) utilizando Inteligencia Artificial. Este modelo no solo anticipa las decisiones de los equipos rivales, sino que demuestra cómo el Deep Learning puede aplicarse a entornos de alta incertidumbre y datos en tiempo real para optimizar la toma de decisiones estratégicas.

## **Stack Tecnológico**
* **Lenguajes**: Python

* **Extracción de Datos**: OpenF1 API

* **Análisis y Manipulación**: Pandas, NumPy

* **Deep Learning**: PyTorch (Arquitectura de Redes Neuronales LSTM)

* **Interpretabilidad (XAI)**: SHAP (SHapley Additive exPlanations)

## **Fases del Proyecto**
1. **Recopilación de Datos**: Consumo de la API de OpenF1 para extraer telemetría histórica, tiempos por vuelta, tipos de neumáticos y datos meteorológicos.

2. **Feature Engineering & EDA**: Limpieza de datos continuos y discretos, y creación de variables temporales clave para el entrenamiento de redes secuenciales.

3. **Modelado Avanzado (Deep Learning)**: Diseño, entrenamiento y validación de una Red Neuronal Recurrente con celdas LSTM (Long Short-Term Memory) mediante PyTorch, ideal para capturar la naturaleza secuencial de las vueltas de una carrera.

4. **Explicabilidad Algorítmica**: Aplicación de valores SHAP para interpretar las predicciones del modelo. Esto permite entender el peso exacto de cada variable (ej. desgaste del neumático vs. tiempo del rival) en la decisión final, eliminando el efecto "caja negra" de la red neuronal.

## **Resultados Clave**
El modelo LSTM es capaz de procesar secuencias temporales complejas para anticipar estrategias de carrera. La integración de SHAP garantiza que los resultados sean interpretables desde el punto de vista del negocio, permitiendo a los ingenieros de pista confiar en la predicción algorítmica.

Este proyecto corresponde al Trabajo de Fin de Máster (TFM) en Data Science.
