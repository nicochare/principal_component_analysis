import numpy as np

def creaPrototipos(nuevaBase: np.ndarray, A: np.ndarray, YTrain: np.ndarray) -> np.ndarray:
    w = nuevaBase.T @ A

    clases = np.unique(YTrain)
    n_clases = np.zeros_like(clases)

    prototipos = np.zeros(shape=(w.shape[0], clases.size))

    for i in range(w.shape[1]):
        clase = np.where(clases == YTrain[i])[0]
        n_clases[clase] += 1
        prototipos[:, clase] += w[:, i].reshape(prototipos[:, clase].shape)

    prototipos /= n_clases

    return prototipos
