import numpy as np
import os
import pandas as pd

import modelcomp as mcp


def read_data(read_from_unjoined):
    """
    Imports data from the filesystem
    :param read_from_unjoined:
    :return: interpolation of tprs, fprs, tprs, interpolation of recalls, precisions, recalls, aucs, pr aucs, feature importances and shap values)
    """
    save_to = mcp.join_save(os.path.join(read_from_unjoined, 'data'))
    interp_tpr = []
    interp_recall = []
    fprs = []
    tprs = []
    precisions = []
    recalls = []
    count = 0
    while os.path.exists(os.path.join(save_to, 'interp_tpr' + str(count) + '.csv')):
        interp_tpr.append(np.genfromtxt(os.path.join(save_to, 'interp_tpr' + str(count) + '.csv'), delimiter=','))
        count += 1
    count = 0
    while os.path.exists(os.path.join(save_to, 'interp_recall' + str(count) + '.csv')):
        interp_recall.append(np.genfromtxt(os.path.join(save_to, 'interp_recall' + str(count) + '.csv'), delimiter=','))
        count += 1
    count = 0
    aucs = np.genfromtxt(os.path.join(save_to, 'aucs.csv'), delimiter=',')
    pr_aucs = np.genfromtxt(os.path.join(save_to, 'pr_aucs.csv'), delimiter=',')
    if os.path.exists(os.path.join(save_to, 'feature_importance.csv')):
        feature_importances = pd.read_csv(os.path.join(save_to, 'feature_importance.csv'), index_col=0)
    else:
        feature_importances = None
    if os.path.exists(os.path.join(save_to, 'shap_values.csv')):
        shap_values = pd.read_csv(os.path.join(save_to, 'shap_values.csv'), index_col=0)
    else:
        shap_values = None
    while os.path.exists(os.path.join(save_to, 'fpr' + str(count) + '.csv')):
        fprs.append(np.genfromtxt(os.path.join(save_to, 'fpr' + str(count) + '.csv'), delimiter=','))
        count += 1
    count = 0
    while os.path.exists(os.path.join(save_to, 'tpr' + str(count) + '.csv')):
        tprs.append(np.genfromtxt(os.path.join(save_to, 'tpr' + str(count) + '.csv'), delimiter=','))
        count += 1
    count = 0
    while os.path.exists(os.path.join(save_to, 'precision' + str(count) + '.csv')):
        precisions.append(np.genfromtxt(os.path.join(save_to, 'precision' + str(count) + '.csv'), delimiter=','))
        count += 1
    count = 0
    while os.path.exists(os.path.join(save_to, 'recall' + str(count) + '.csv')):
        recalls.append(np.genfromtxt(os.path.join(save_to, 'recall' + str(count) + '.csv'), delimiter=','))
        count += 1
    for list_data in [interp_tpr, interp_recall, fprs, tprs, precisions, recalls]:
        if len(list_data) == 0:
            list_data = None
        elif len(list_data) == 1:
            list_data = list_data[0]
    for array_data in [aucs, pr_aucs]:
        if len(array_data.shape) == 0: array_data = array_data.item()
    return interp_tpr, fprs, tprs, interp_recall, precisions, recalls, aucs, pr_aucs, feature_importances, shap_values
