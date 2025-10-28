from scipy.io import loadmat

def loadImages(directory:str):

    data  = loadmat(directory)

    train_images = data['dataset'][0][0][0][0][0][0]
    train_labels = data['dataset'][0][0][0][0][0][1].flatten()

    test_images = data['dataset'][0][0][1][0][0][0]
    test_labels = data['dataset'][0][0][1][0][0][1].flatten()

    return train_images, train_labels, test_images, test_labels

