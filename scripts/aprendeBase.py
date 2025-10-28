import numpy as np

def reshape_imgs(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[1]*x.shape[2])
    
    R = x.T # Imagen por columna
    return R

def aprendeBase(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(X, list):
        X = np.asarray(X)

    R = reshape_imgs(X)

    media = np.mean(R, axis=1, keepdims=True)

    A = R-media

    C = A.T @ A

    u, v = np.linalg.eigh(C)

    with np.errstate(invalid='ignore'): # Evitar warnings por NaN y division por 0
        sqrt_u = np.sqrt(u)
        nuevaBase = (A @ v) / sqrt_u

    nuevaBase[np.isnan(nuevaBase)] = 0
    nuevaBase[np.isinf(nuevaBase)] = 0
    
    return media, A, nuevaBase.astype(np.float64)