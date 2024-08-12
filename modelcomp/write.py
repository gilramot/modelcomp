import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import modelcomp as mcp


def write_data(save_to_unjoined, label=None, interp_tpr=None, interp_recall=None, aucs=None, pr_aucs=None, fprs=None,
               tprs=None,
               precisions=None, recalls=None, feature_importances=None, shap_values=None):
    """
    Writes data to the filesystem
    :param save_to_unjoined: location to save to
    :param label: label names
    :param interp_tpr: interpolation of tprs
    :param interp_recall: interpolation of recalls
    :param aucs: aucs
    :param pr_aucs: pr aucs
    :param fprs: fprs
    :param tprs: tprs
    :param precisions: precisions
    :param recalls: recalls
    :param feature_importances: feature importances
    :param shap_values: shap values
    :return: data exported to the filesystem (default: "export/<train label>/<test label>/<model name>/data")
    """
    save_to = os.path.join('export', save_to_unjoined, 'data')
    mcp.make_dir(save_to)
    if interp_tpr is not None:
        for index, value in enumerate([interp_tpr] if type(interp_tpr) is np.ndarray else interp_tpr):
            np.savetxt(os.path.join(save_to, 'interp_tpr' + str(index) + '.csv'), value, delimiter=',')
    if interp_recall is not None:
        for index, value in enumerate([interp_recall] if type(interp_recall) is np.ndarray else interp_recall):
            np.savetxt(os.path.join(save_to, 'interp_recall' + str(index) + '.csv'), value, delimiter=',')
    if aucs is not None:
        np.savetxt(os.path.join(save_to, 'aucs.csv'), aucs, delimiter=',')
    if pr_aucs is not None:
        np.savetxt(os.path.join(save_to, 'pr_aucs.csv'), pr_aucs, delimiter=',')
    if feature_importances is not None:
        feature_importances.to_csv(os.path.join(save_to, 'feature_importance.csv'))
    if shap_values is not None:
        shap_columns = pd.DataFrame(shap_values, columns=label)
        vals = np.abs(shap_columns.values).mean(0)
        avg_shap = pd.DataFrame(list(zip(label, vals)),
                                columns=['col_name', 'avg_shap_value'])
        avg_shap.sort_values(by=['avg_shap_value'],
                             ascending=False, inplace=True)
        avg_shap.to_csv(os.path.join(save_to, 'shap_values.csv'), index=False)
    if fprs is not None:
        for index, value in enumerate(fprs):
            np.savetxt(os.path.join(save_to, 'fpr' + str(index) + '.csv'), value, delimiter=',')
    if tprs is not None:
        for index, value in enumerate(tprs):
            np.savetxt(os.path.join(save_to, 'tpr' + str(index) + '.csv'), value, delimiter=',')
    if precisions is not None:
        for index, value in enumerate(precisions):
            np.savetxt(os.path.join(save_to, 'precision' + str(index) + '.csv'), value, delimiter=',')
    if recalls is not None:
        for index, value in enumerate(recalls):
            np.savetxt(os.path.join(save_to, 'recall' + str(index) + '.csv'), value, delimiter=',')


def write_plot(save_to, img_name):
    """
    Export the current plt state to the filesystem
    :param save_to: location to save to
    :param img_name: desired file name
    :return: the current plt state exported to the filesystem
    """
    dir = save_to
    mcp.make_dir(dir)
    plt.savefig(os.path.join(dir, img_name + '.png'), bbox_inches='tight')
    # saving plot
    plt.clf()
