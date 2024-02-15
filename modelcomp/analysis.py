import numpy as np
import pandas as pd
import shap
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import modelcomp as mcp


def get_fprtpr(model, X, y, pos_num):
    """
    Calculates the fpr & tpr of a model based on the train & test data
    :param model: The trained model
    :param X: Testing data
    :param y: Binary labels
    :param pos_num: Positive class label
    :return: fpr & tpr values
    """
    fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1], pos_label=pos_num, drop_intermediate=False)
    return fpr, tpr
    # returning fpr, tpr of roc_curve


def get_pr(model, X, y, pos_num):
    """
    Calculates the precision & recall of a model based on the train & test data
    :param model: The trained model
    :param X: Testing data
    :param y: Binary labels
    :param pos_num: Positive class label
    :return: precision & recall values
    """
    precision, recall, _ = precision_recall_curve(y, model.predict_proba(X)[:, 1], pos_label=pos_num)
    return precision, recall
    # returning precision, recall of rp_curve


def std_validation_models(models, X_train, X_test, y_train, y_test, tested_on, trained_on, feature_names, validate=True,
                          explain=True, plot=True):
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
    :param plot: Plot the results (default: True)
    :return: Model results, exported to the filesystem (default: "export")
    """
    X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy().ravel(), y_test.to_numpy().ravel()
    mean_fpr = np.linspace(0, 1, 100)
    # init
    for model_index, model in enumerate(models):
        X_train_temp, X_test_temp = X_train, X_test
        interp_tpr, interp_recall, aucs, pr_aucs = None, None, None, None
        feature_importances, shap_values = None, None
        save_to_unjoined = mcp.data_to_filename(tested_on, mcp.model_names[model_index],
                                                trained_on=trained_on)
        # in-loop init
        if validate:
            aucs = []
            pr_aucs = []
            if type(model) is SVC:
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
            # scaling if svm
            model.fit(X_train, y_train)
            # fitting model
            fpr, tpr = get_fprtpr(model, X_test, y_test, 1)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            aucs.append(metrics.auc(fpr, tpr))
            # roc curve variables
            precision, recall = get_pr(model, X_test, y_test.ravel(), 1)
            interp_recall = np.interp(mean_fpr, recall[::-1].ravel(), precision[::-1].ravel())
            interp_recall[0] = 1.0
            pr_aucs.append(metrics.auc(recall, precision))
            # pr curve variables
        if explain:
            feature_importances = pd.DataFrame(mcp.get_feature_importance(model).T, index=feature_names,
                                               columns=['Importance']) if type(
                model) is not KNeighborsClassifier else None
            # feature importance
            shap_values = (shap.explainers.Permutation(model.predict, X_test, max_evals=1000).shap_values(X_test))
            # shap values
        mcp.write_data(save_to_unjoined, feature_names, interp_tpr, interp_recall, aucs, pr_aucs,
                       feature_importances=feature_importances, shap_values=shap_values)
        # exporting data
        X_train, X_test = X_train_temp, X_test_temp
        if plot:
            mcp.individual_plots(save_to_unjoined)
        # plotting data


def cross_val_models(models, validation_model, X, y, positive_label, feature_names, validate=True,
                     explain=True, plot=True):
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
    :return: Model results, exported to the filesystem (default: "export")
    """
    mean_fpr = np.linspace(0, 1, 100)
    # init
    for model_index, model in enumerate(models):
        feature_importances, shap_values = None, None
        if validate:
            feature_importances_per_fold = []
            interp_tpr_per_fold = []
            aucs = []
            interp_recall_per_fold = []
            pr_aucs = []
            fprs = []
            tprs = []
            precisions = []
            recalls = []
            shap_values = None
            for split_index, (train, test) in enumerate(validation_model.split(X, y)):
                X_train_temp, X_test_temp = X[train], X[test]
                if type(model) is SVC:
                    sc = StandardScaler()
                    X[train] = sc.fit_transform(X[train])
                    X[test] = sc.transform(X[test])
                    # scaling if svm
                model.fit(X[train], y[train])
                # fitting model
                fpr, tpr = get_fprtpr(model, X[test], y[test], 1)
                fprs.append(fpr)
                tprs.append(tpr)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tpr_per_fold.append(interp_tpr)
                aucs.append(metrics.auc(fpr, tpr))
                # roc curve variables
                precision, recall = get_pr(model, X[test], y[test].ravel(), 1)
                precisions.append(precision)
                recalls.append(recall)
                interp_recall = np.interp(mean_fpr, recall[::-1].ravel(), precision[::-1].ravel())
                interp_recall[0] = 1.0
                interp_recall_per_fold.append(interp_recall)
                pr_aucs.append(metrics.auc(recall, precision))
                # pr curve variables
                if explain:
                    feature_importances_per_fold.append(
                        pd.DataFrame(mcp.get_feature_importance(model).T, index=feature_names,
                                     columns=['Importance']) if type(model) is not KNeighborsClassifier else None)
                    if feature_importances_per_fold[0] is None:
                        feature_importances = None
                    else:
                        feature_importances = feature_importances_per_fold[0]
                        for feature_importances_in_fold in feature_importances_per_fold[:1]:
                            feature_importances = feature_importances.add(feature_importances_in_fold, fill_value=0)
                        feature_importances['Importance'] = feature_importances['Importance'].map(
                            lambda old_value: old_value / len(feature_importances_per_fold))
                    # feature importance
                    shap_values_temp = shap.explainers.Permutation(model.predict, X[test], max_evals=1000).shap_values(
                        X[test])
                    if shap_values is None:
                        shap_values = shap_values_temp
                    else:
                        shap_values = np.append(shap_values, shap_values_temp, axis=0)
                    # shap values
                X[train], X[test] = X_train_temp, X_test_temp
            mcp.write_data(
                mcp.data_to_filename(positive_label, mcp.model_names[model_index]),
                feature_names,
                interp_tpr_per_fold, interp_recall_per_fold, aucs, pr_aucs, fprs=fprs, tprs=tprs,
                precisions=precisions, recalls=recalls, feature_importances=feature_importances,
                shap_values=shap_values)
            # exporting data
        if plot:
            mcp.individual_plots(
                mcp.data_to_filename(positive_label, mcp.model_names[model_index]))
        # plotting data
