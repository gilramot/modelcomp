from sklearn.neighbors import KNeighborsClassifier


def get():
    """
    Load a k-NN model from the scikit-learn library
    :return: a k-NN model
    """
    return KNeighborsClassifier()
