from scripts.loadImages import loadImages
from scripts.get_n_samples_per_class import divide_data
from scripts.clasificar import clasificacion 
import numpy as np

def calcular(XTrain: np.ndarray, YTrain: np.ndarray, XTest: np.ndarray, YTest: np.ndarray, clases: list, cant_datos: int) -> float:
    porcentajeAciertos = clasificacion(XTrain, YTrain, XTest, YTest)

    print(f">>> El porcentaje de aciertos de\n- {int(cant_datos)} TOTAL\n- {int(cant_datos*0.8)} TRAIN\n- {int(cant_datos*0.2)} TEST\n- Clases {clases} es →", "[{:.2f}%]\n".format(porcentajeAciertos))

def main():
    data_route = "data"
    
    XTrain, YTrain, XTest, YTest = loadImages(data_route)

    cant_datos = 100

    # Clasificacion 1 y 7
    clases = [1, 7]
    XTrain_1, YTrain_1, XTest_1, YTest_1 = divide_data(XTrain, YTrain, XTest, YTest, clases, cant_datos)
    calcular(XTrain_1, YTrain_1, XTest_1, YTest_1, clases, cant_datos)

    # Clasificacion 2 y 7
    clases = [2, 7]
    XTrain_2, YTrain_2, XTest_2, YTest_2 = divide_data(XTrain, YTrain, XTest, YTest, clases, cant_datos)
    calcular(XTrain_2, YTrain_2, XTest_2, YTest_2, clases, cant_datos)
    
    # Clasificacion 4 y 9
    clases = [4, 9]
    XTrain_3, YTrain_3, XTest_3, YTest_3 = divide_data(XTrain, YTrain, XTest, YTest, clases, cant_datos)
    calcular(XTrain_3, YTrain_3, XTest_3, YTest_3, clases, cant_datos)

    # Codigo utilizado para obtener los datos de la figura 2
    # for cant in range(100, 7000, 500):
    #     clases = [4, 9]
    #     XTrain_3, YTrain_3, XTest_3, YTest_3 = divide_data(XTrain, YTrain, XTest, YTest, clases, cant)
    #     calcular(XTrain_3, YTrain_3, XTest_3, YTest_3, clases, cant)

if __name__ == "__main__":
    main()