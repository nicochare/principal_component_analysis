from mnist import MNIST
import numpy as np
import os
# pip install python-mnist

def loadImages(directory):
    mndata = MNIST(os.getcwd() + "\\" +directory)
    XTrain, YTrain = mndata.load_training()
    XTest, YTest = mndata.load_testing()

    return np.array(XTrain), np.array(YTrain), np.array(XTest), np.array(YTest)