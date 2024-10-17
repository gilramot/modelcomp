from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)
import warnings
import numpy as np

__all__ = [
    "METRICS",
    "SCORE_METRICS",
    "CURVE_METRICS",
    "IMPORTANCE_METRICS",
    "shap_explainer",
    "builtin_importance",
]



def shap_explainer(model, test, X_test, random_state=None):
    """
    Gets the SHAP explainer of a trained model
    :param model: The trained model
    :param test: The test data indices
    :param X_test: The test data
    :return:
    """
    try:
        import shap
    except ImportError:
        raise ImportError(f"shap is required for feature importance calculation")
    max_evals = 500
    if X_test.shape[0] >= 250:
        max_evals = X_test.shape[1] * 2 + 1
        warnings.warn(
            f"The number of samples in the test set is greater than 250. max_evals updated to {max_evals}")
    try:
        explainer = shap.Explainer(model, seed=random_state)
    except TypeError:
        warnings.warn(
            "Model is not callable by the SHAP Explainer. Using model.predict instead"
        )
        explainer = shap.Explainer(model.predict, X_test, max_evals=max_evals, seed=random_state)
    shap_values = explainer(X_test)
    return shap_values


def builtin_importance(model, attr=None):
    """
    Gets the feature importance of a trained model
    :param model: The trained model (to get a return value needs to have a known attribute or a custom attribute name)
    :param attr: The model's attribute name (default: None)
    :return:
    """
    if attr is not None:
        if hasattr(model, attr):
            try:
                return getattr(model, attr)
            except AttributeError:
                try:
                    return getattr(model, attr)()
                except ValueError:
                    ValueError(
                        f"{model.__class__.__name__}.{attr}() return value is not of a numerical type"
                    )
            except ValueError:
                raise ValueError(
                    f"{model.__class__.__name__}.{attr} is not of a numerical type"
                )

        else:
            raise AttributeError(
                f"{model.__class__.__name__} has no attribute named {attr}"
            )

    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        return model.coef_

    warnings.warn(
        f"{model.__class__.__name__} has no known attribute and no custom attribute was given, so None was returned"
    )
    return None


SCORE_METRICS = {
    "accuracy_score": accuracy_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "f1_score": f1_score,
    "roc_auc_score": roc_auc_score,
    "average_precision_score": average_precision_score,
}

CURVE_METRICS = {
    "precision_recall_curve": precision_recall_curve,
    "roc_curve": roc_curve,
}

IMPORTANCE_METRICS = {
    "builtin_importance": builtin_importance,
    "shap_explainer": shap_explainer,
}

METRICS = {**SCORE_METRICS, **CURVE_METRICS, **IMPORTANCE_METRICS}
