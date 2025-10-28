from scripts.loadImages_EMNIST import loadImages
from scripts.filter_clases import obtener_datos
from scripts.clasificar import clasificacion
import numpy as np

def calcular(XTrain: np.ndarray, YTrain: np.ndarray, XTest: np.ndarray, YTest: np.ndarray, clases: np.ndarray, cant_datos: int) -> float:
    porcentajeAciertos = clasificacion(XTrain, YTrain, XTest, YTest)
    
    train_ex = int(cant_datos * 26)
    test_ex = int((cant_datos//10)*26)
    total = train_ex + test_ex

    print(f">>> El porcentaje de aciertos de\n- {total} TOTAL\n- {train_ex} TRAIN\n- {test_ex} TEST\n- Clases {clases[0]} a {clases[-1]} es →", "[{:.2f}%]\n".format(porcentajeAciertos))

    return porcentajeAciertos

def main():
    data_route = "data/emnist-letters.mat"
    xtr, ytr, xte, yte = loadImages(data_route)

    # 26 clases
    clases = np.arange(1, 27)

    # [1144, 1716, 2288, 2860, 3432, 5720, 8008, 11440] total de ejemplos en cada uno
    sz = [40, 60, 80, 100, 120, 200, 280, 400] # para train, por cada clase

    for cant in sz:
        XTrain, YTrain, XTest, YTest = obtener_datos(xtr, ytr, xte, yte, cant)
        calcular(XTrain, YTrain, XTest, YTest, clases, cant)
    
if __name__ == "__main__":
    main()