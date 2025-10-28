import numpy as np
from scripts.creaPrototipos import creaPrototipos
from scripts.aprendeBase import aprendeBase

def clasificar(nuevaBase: np.ndarray, media: np.ndarray, prototipos: np.ndarray, imagenes: np.ndarray, clasesReales: np.ndarray) -> np.ndarray:
    if imagenes.ndim == 3:
        imagenes = imagenes.reshape(imagenes.shape[0], -1)
    imgs_col = imagenes.T

    w = (nuevaBase.T @ (imgs_col - media)).T

    clases = np.unique(clasesReales)

    aciertos = 0
    for i in range(w.shape[0]):
        distancia = np.sum(np.square(w[i, None].T - prototipos), axis=0)

        claseMin = np.argmin(distancia)

        if clasesReales[i] == clases[claseMin]:
            aciertos += 1

    porcentajeAciertos = (aciertos/w.shape[0])*100
    return porcentajeAciertos

def clasificacion(XTrain: np.ndarray, YTrain: np.ndarray, XTest: np.ndarray, YTest: np.ndarray):
    media, A, nuevaBase = aprendeBase(XTrain)

    prototipos = creaPrototipos(nuevaBase, A, YTrain)

    return clasificar(nuevaBase, media, prototipos, XTest, YTest)