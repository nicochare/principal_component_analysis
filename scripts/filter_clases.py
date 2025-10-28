import numpy as np

def filterClasses(train_images, train_labels, test_images, test_labels, N_EX_TRAIN, N_EX_TEST):
    unique_clases = np.unique(train_labels).astype(int)
    img_per_class_train = int(N_EX_TRAIN/len(unique_clases))
    ima_per_class_test = int(N_EX_TEST/len(unique_clases))

    XTrain = np.zeros((N_EX_TRAIN, train_images.shape[1]))
    YTrain = np.zeros((N_EX_TRAIN))

    XTest = np.zeros((len(unique_clases) * ima_per_class_test, train_images.shape[1]))
    YTest = np.zeros((len(unique_clases) * ima_per_class_test))

    for i in unique_clases:
        train_idx = np.where(train_labels == i)
        XTrain[(i-1)*img_per_class_train: i*img_per_class_train] = train_images[train_idx,:][0][:img_per_class_train,:]
        YTrain[(i-1)*img_per_class_train: i*img_per_class_train] = train_labels[train_idx][:img_per_class_train]

        test_idx = np.where(test_labels == i)
        XTest[(i-1)*ima_per_class_test: i*ima_per_class_test] = test_images[test_idx,:][0][:ima_per_class_test,:]
        YTest[(i-1)*ima_per_class_test: i*ima_per_class_test] = test_labels[test_idx][:ima_per_class_test]

    XTrain = np.transpose(XTrain.reshape(N_EX_TRAIN,28,28), (0, 2, 1)) # Las imagenes estan rotadas en el dataset original
    XTest = np.transpose(XTest.reshape(N_EX_TEST,28,28), (0, 2, 1)) # Las imagenes estan rotadas en el dataset original

    return XTrain, YTrain, XTest, YTest

def obtener_datos(train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray, test_labels: np.ndarray, cant_datos: int) -> float:
    N_EX_TRAIN = int(cant_datos * 26)
    N_EX_TEST = int((cant_datos/10)*26)

    XTrain, YTrain, XTest, YTest = filterClasses(train_images, train_labels, test_images, test_labels, N_EX_TRAIN, N_EX_TEST)
    
    return XTrain, YTrain, XTest, YTest