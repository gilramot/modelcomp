from sklearn.svm import SVC


def get(seed=None):
    """
    Load an svm model from the scikit-learn library
    :param seed: set seed for model randomness (default: None)
    :return: an svm model
    """
    return SVC(random_state=(seed if seed is not None else None), kernel='linear', probability=True)
