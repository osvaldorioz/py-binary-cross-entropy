La **Entropía Cruzada Binaria** (Binary Cross-Entropy, BCE) es una función de pérdida utilizada en problemas de clasificación binaria. Mide la discrepancia entre las predicciones de un modelo y las etiquetas reales, evaluando qué tan bien el modelo predice probabilidades cercanas a los valores esperados (0 o 1). Matemáticamente, se define como:

![image](https://github.com/user-attachments/assets/0bc63c29-cc83-4b3e-aac9-52f9874bca36)


Para implementar esta función de pérdida en C++ utilizando PyTorch y Pybind11, se siguieron los siguientes pasos:

1. **Implementación en C++:**
   - Se creó una función que recibe tensores de entrada y objetivo, aplica la función sigmoide a las entradas y luego calcula la pérdida de entropía cruzada binaria utilizando la función `torch::binary_cross_entropy`.

2. **Integración con Pybind11:**
   - Se utilizó Pybind11 para exponer la función C++ como un módulo que puede ser importado y utilizado en Python.

3. **Uso en Python:**
   - Desde un script de Python, se importó el módulo compilado y se utilizó la función implementada para calcular la pérdida sobre datos de ejemplo.
   - Además, se generó una gráfica que muestra los datos y la función sigmoide para visualizar el comportamiento del modelo.

Este enfoque combina la eficiencia del cálculo en C++ con la flexibilidad de Python para la manipulación y visualización de datos. 
