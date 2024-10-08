import numpy as np
import pandas as pd
import shap
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import modelcomp as mc
from tqdm import tqdm

__all__ = [
    "cross_val_models",
    "std_validation_models",
    "get_fprtprauc",
    "get_pr",
]


def get_weight(accuracy, auc, pr_auc):
    weight = max(auc - 0.5, 0) + max(pr_auc - 0.5, 0) + max(accuracy - 0.5, 0)
    return 1 if weight == 0 else weight


def scale_train_and_test(train=None, test=None):
    sc = StandardScaler()
    train = sc.fit_transform(train)
    test = sc.transform(test)
    return train, test


def get_accuracy(model, X, y):
    return accuracy_score(y, model.predict(X))


def get_fprtprauc(model, X, y):
    """
    Calculates the fpr, tpr & auc of a model based on the train & test data
    :param model: The trained model
    :param X: Testing data
    :param y: Binary labels
    :return: fpr, tpr values
    """
    mean_fpr = np.linspace(0, 1, 100)
    n_classes = list(set(y.tolist()))
    pass_index = 0
    if len(n_classes) == 2:
        n_classes.pop(0)
        pass_index += 1
    fpr = []
    tpr = []
    aucs = []
    for class_idx, class_name in enumerate(n_classes):
        fpr_temp, tpr_temp, _ = roc_curve(
            np.where(y == class_name, 1, 0),
            model.predict_proba(X)[:, class_idx + pass_index],
        )
        fpr.append(fpr_temp)
        tpr.append(tpr_temp)
        aucs.append(metrics.auc(fpr_temp, tpr_temp))
    interp = np.zeros_like(mean_fpr)
    for i in range(len(n_classes)):
        interp = interp + np.interp(mean_fpr, fpr[i], tpr[i])
    interp = interp / len(n_classes)
    interp[0] = 0.0
    return interp, np.mean(aucs)
    # returning fpr, tpr of roc_curve


def get_pr(model, X, y):
    """
    Calculates the precision & recall of a model for multiclass classification
    :param model: The trained model
    :param X: Testing data
    :param y: True labels
    :return: precision & recall values for each class
    """
    mean_recall = np.linspace(0, 1, 100)
    n_classes = list(set(y.tolist()))
    pass_index = 0
    if len(n_classes) == 2:
        n_classes.pop(0)
        pass_index += 1
    recall = []
    precision = []
    pr_aucs = []
    for class_idx, class_name in enumerate(n_classes):
        precision_temp, recall_temp, _ = precision_recall_curve(
            np.where(y == class_name, 1, 0),
            model.predict_proba(X)[:, class_idx + pass_index],
        )
        pr_aucs.append(metrics.auc(recall_temp, precision_temp))
        recall.append(recall_temp)
        precision.append(precision_temp)
    interp = np.zeros_like(mean_recall)
    for i in range(len(n_classes)):
        interp = interp + np.interp(mean_recall, precision[i], recall[i])
    interp = interp / len(n_classes)
    interp[0] = 1.0
    return interp, np.mean(pr_aucs)
    # returning recall, precision of roc_curve


def get_accuracy_metrics(model, X_test, y_test):
    accuracy = get_accuracy(model, X_test, y_test)
    # accuracy calculation
    interp_tprs, auc = get_fprtprauc(model, X_test, y_test)
    # roc curve variables
    interp_recalls, pr_auc = get_pr(model, X_test, y_test)
    return accuracy, interp_tprs, auc, interp_recalls, pr_auc


def get_explainers(model, X_test, feature_names):
    feature_importances = (
        pd.DataFrame(
            mc.utilities.get_feature_importance(model).T,
            index=feature_names,
            columns=["Importance"],
        )
        if type(model) is not KNeighborsClassifier
        and type(model) is not VotingClassifier
        else None
    )
    # feature importance
    shap_values = shap.explainers.Permutation(
        model.predict, X_test, max_evals=1000
    ).shap_values(X_test)
    # shap values
    return feature_importances, shap_values

def std_validation_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test,
    trained_on,
    tested_on,
    feature_names,
    validate=True,
    explain=True,
    write=False,
    plot=True,
):
    """
    Standard validation of models (used when the positive class label differs between the training and testing data)
    :param models: A list of models to evaluate
    :param X_train: Training data
    :param X_test: Testing data
    :param y_train: Training data labels
    :param y_test: Testing data labels
    :param tested_on: Tested label
    :param trained_on: Trained label
    :param feature_names: List of feature names
    :param validate: Validate the models (default: True)
    :param explain: Explain the models (default: True)
    :param write: Write the results to the filesystem (default: False)
    :param plot: Plot the results (default: True)
    :return: Model results, exported to the filesystem (default: 'export')
    """
    X_train, X_test, y_train, y_test = (
        X_train.to_numpy(),
        X_test.to_numpy(),
        y_train.to_numpy().ravel(),
        y_test.to_numpy().ravel(),
    )
    # init
    model_results = []
    for model_index, model in enumerate(models):
        X_train_temp, X_test_temp = X_train, X_test
        interp_tpr, interp_recall, aucs, pr_aucs, accuracies = (
            None,
            None,
            None,
            None,
            None,
        )
        feature_importances, shap_values = None, None
        save_to_unjoined = mc.utilities.data_to_filename(
            tested_on, model.__class__.__name__, trained_on=trained_on
        )
        # in-loop init
        if validate:
            accuracies = []
            aucs = []
            pr_aucs = []
            if type(model) is SVC or type(model) is LogisticRegression:
                X_train, X_test = scale_train_and_test(X_train, X_test)
            # scaling if svm
            model.fit(X_train, y_train)
            # fitting model
            accuracy, interp_tpr, auc, interp_recall, pr_auc = get_accuracy_metrics(
                model, X_test, y_test
            )
            accuracies.append(accuracy)
            aucs.append(auc)
            pr_aucs.append(pr_auc)
            # obtaining accuracy metrics
            if explain:
                feature_importances, shap_values = get_explainers(
                    model, X_test, feature_names
                )
        if write:
            mc.write.write_data(
                save_to_unjoined,
                accuracies,
                interp_tpr,
                interp_recall,
                aucs,
                pr_aucs,
                feature_importances=feature_importances,
                shap_values=shap_values,
                label=feature_names,
            )
            # exporting data
        X_train, X_test = X_train_temp, X_test_temp
        if plot:
            mc.plotter.individual_plots(
                save_to_unjoined,
                accuracies=accuracies,
                interp_tpr=interp_tpr,
                interp_recall=interp_recall,
                aucs=aucs,
                pr_aucs=pr_aucs,
                feature_importances=feature_importances,
                shap_values=shap_values,
            )
        # plotting data
        model_results.append(
            [
                model.__class__.__name__,
                accuracies,
                interp_tpr,
                interp_recall,
                aucs,
                pr_aucs,
                feature_importances,
                shap_values,
            ]
        )
    return model_results


def cross_val_models(
    models,
    validation_model,
    X,
    y,
    positive_label,
    feature_names,
    validate=True,
    explain=True,
    plot=True,
    write=False,
):
    """
    Cross validation of multiple models
    :param models: List of models to evaluate
    :param validation_model: Validation model to evaluate with
    :param X: Features
    :param y: Labels
    :param positive_label: Positive label
    :param feature_names: List of feature names
    :param validate: Validate the models (default: True)
    :param explain: Explain the models (default: True)
    :param plot: Plot the results (default: True)
    :param write: Write the results to the filesystem (default: False)
    :return: Model results, exported to the filesystem (default: 'export')
    """
    # init
    model_results = []
    for model_index, model in tqdm(enumerate(models), desc="Model"):
        feature_importances, shap_values = None, None
        if validate:
            feature_importances_per_fold = []
            interp_tpr_per_fold = []
            accuracies = []
            aucs = []
            interp_recall_per_fold = []
            pr_aucs = []
            shap_values = None
            for split_index, (train, test) in enumerate(
                list(validation_model.split(X, y))
            ):
                X_train_temp, X_test_temp = X[train], X[test]
                if type(model) is SVC or type(model) is LogisticRegression:
                    X[train], X[test] = scale_train_and_test(X[train], X[test])
                model.fit(X[train], y[train])
                # fitting model
                accuracy, interp_tpr, auc, interp_recall, pr_auc = get_accuracy_metrics(
                    model, X[test], y[test]
                )
                accuracies.append(accuracy)
                interp_tpr_per_fold.append(interp_tpr)
                aucs.append(auc)
                interp_recall_per_fold.append(interp_recall)
                pr_aucs.append(pr_auc)
                # obtaining accuracy metrics
                if explain:
                    feature_importances_temp, shap_values_temp = get_explainers(
                        model, X[test], feature_names
                    )
                    feature_importances_per_fold.append(feature_importances_temp)
                    if feature_importances_per_fold[0] is None:
                        feature_importances = None
                    else:
                        feature_importances = feature_importances_per_fold[0]
                        for feature_importances_in_fold in feature_importances_per_fold[
                            :1
                        ]:
                            feature_importances = feature_importances.add(
                                feature_importances_in_fold, fill_value=0
                            )
                        feature_importances["Importance"] = feature_importances[
                            "Importance"
                        ].map(
                            lambda old_value: old_value
                            / len(feature_importances_per_fold)
                        )
                    # feature importance
                    if shap_values is None:
                        shap_values = shap_values_temp
                    else:
                        shap_values = np.append(shap_values, shap_values_temp, axis=0)
                    # shap values
                X[train], X[test] = X_train_temp, X_test_temp
            if write:
                mc.write.write_data(
                    mc.utilities.data_to_filename(
                        positive_label, model.__class__.__name__
                    ),
                    accuracies,
                    interp_tpr_per_fold,
                    interp_recall_per_fold,
                    aucs,
                    pr_aucs,
                    feature_importances=feature_importances,
                    shap_values=shap_values,
                    label=feature_names,
                )
                # exporting data
            model_results.append(
                [
                    accuracies,
                    interp_tpr_per_fold,
                    interp_recall_per_fold,
                    aucs,
                    pr_aucs,
                    feature_importances,
                    shap_values,
                ]
            )
        if plot:
            mc.plotter.individual_plots(
                mc.utilities.data_to_filename(positive_label, model.__class__.__name__),
                accuracies=accuracies,
                interp_tpr=interp_tpr_per_fold,
                interp_recall=interp_recall_per_fold,
                aucs=aucs,
                pr_aucs=pr_aucs,
                feature_importances=feature_importances,
                shap_values=shap_values,
            )
        # plotting data
