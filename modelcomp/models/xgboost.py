import xgboost as xgb


def get(seed=None):
    """
    Load an XGBoost model from the xgboost library
    :param seed: set seed for model randomness (default: None)
    :return: an xgboost model
    """
    return xgb.XGBClassifier(random_state=(seed if seed is not None else None))
