from sklearn.ensemble import RandomForestClassifier


def get(seed=None):
    """
    Load a random forest from the scikit-learn library
    :param seed: set seed for model randomness (default: None)
    :return: a random forest model
    """
    return RandomForestClassifier(random_state=(seed if seed is not None else None))
