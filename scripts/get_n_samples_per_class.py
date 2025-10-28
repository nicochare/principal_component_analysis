import random
import numpy as np

def get_n_samples_per_class(X, y, classes, samples_per_class):
    X_sub = np.array([]).reshape(0,len(X[0]))
    Y_sub = np.array([],dtype=int)

    for label in classes:
        random.seed(label)

        X_filter = X[y == label]
        Y_filter = y[y == label]

        random_index = random.sample(range(0,len(X_filter)), int(samples_per_class))

        X_sub = np.vstack([X_sub, X_filter[random_index]])
        Y_sub = np.concatenate([Y_sub, Y_filter[random_index]])

    return X_sub, Y_sub

def divide_data(XTrain: np.ndarray, YTrain: np.ndarray, XTest: np.ndarray, YTest: np.ndarray, clases: list, cant_datos: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_lim = int(cant_datos*0.8)
    test_lim = int(cant_datos*0.2)

    # Normalizamos el conjunto de datos al intervalo [0 ,1]
    XTrain = XTrain / 255
    XTest = XTest / 255
    # Filtramos la dimension y conjunto de clases de interes
    XTrain , YTrain = get_n_samples_per_class(XTrain, YTrain, clases, train_lim / len(clases) )
    XTest , YTest = get_n_samples_per_class(XTest, YTest, clases, test_lim / len (clases) )

    return XTrain, YTrain, XTest, YTest