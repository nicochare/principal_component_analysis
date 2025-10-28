from scripts.loadImages import loadImages
from scripts.get_n_samples_per_class import divide_data
from scripts.clasificar import clasificacion
import numpy as np

def calcular(XTrain: np.ndarray, YTrain: np.ndarray, XTest: np.ndarray, YTest: np.ndarray, clases: list, cant_datos: int) -> float:
    porcentajeAciertos = clasificacion(XTrain, YTrain, XTest, YTest)

    print(f">>> El porcentaje de aciertos de\n- {cant_datos} TOTAL\n- {int(cant_datos*0.8)} TRAIN\n- {int(cant_datos*0.2)} TEST\n- Clases {clases} es →", "[{:.2f}%]\n".format(porcentajeAciertos))

def main():
    clases = [i for i in range(10)]
    data_route = "data"
    
    xtr, ytr, xte, yte = loadImages(data_route)

    for a in range(100, 2000, 200):
        XTrain, YTrain, XTest, YTest = divide_data(xtr, ytr, xte, yte, clases, a)
        calcular(XTrain, YTrain, XTest, YTest, clases, a)

if __name__ == "__main__":
    main()