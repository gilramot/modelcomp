from sklearn.linear_model import LogisticRegression


def get(seed=None):
    """
    Load a logistic regression model from the scikit-learn library
    :param seed: set seed for model randomness (default: None)
    :return: a logistic regression model
    """
    return LogisticRegression(random_state=(seed if seed is not None else None), max_iter=10000)
