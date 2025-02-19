from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import torch
import bce_module 
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/binary-cross-entropy")
def calculo(samples: int, features: int):
    output_file = 'binary-cross-entropy.png'
    
    # Generar datos de ejemplo
    np.random.seed(0)
    data = np.random.rand(samples, features).astype(np.float32)
    # Generar etiquetas binarias (0 o 1) basadas en un umbral
    labels = (data > 0.5).astype(np.float32)

    # Convertir los datos a tensores de PyTorch
    inputs = torch.from_numpy(data)
    targets = torch.from_numpy(labels)

    # Calcular la pérdida de Binary Cross-Entropy utilizando el módulo C++
    loss = bce_module.binary_cross_entropy(inputs, targets)
    #print(f'Pérdida de Binary Cross-Entropy: {loss.item()}')

    # Generar una gráfica para visualizar los datos y la función sigmoide
    plt.figure(figsize=(10, 6))
    plt.scatter(data, labels, label='Datos originales', color='blue')
    plt.plot(np.linspace(0, 1, 100), 1 / (1 + np.exp(-np.linspace(0, 1, 100))), label='Función Sigmoide', color='red')
    plt.xlabel('Entrada')
    plt.ylabel('Salida')
    plt.title('Datos y Función Sigmoide')
    plt.legend()
    plt.grid()
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Pérdida de Binary Cross-Entropy": loss.item(),
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/binary-cross-entropy-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)
