import os
import random
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    precision_recall_curve,
)
import shap
from modelcomp._constants import METRICS
import pickle as pkl
from sklearn.base import clone

__all__ = [
    "remove_falsy_columns",
    "split_data",
    "split_array",
    "get_feature_importance",
    "merge_dfs",
    "relative_abundance",
    "get_label_indexes",
    "filter_data",
    "remove_rare_features",
    "remove_string_columns",
    "make_dir",
    "join_save",
    "data_to_filename",
    "filename_to_data",
    "encode",
]


def remove_falsy_columns(data):
    """
    Remove columns that contain all false values from df
    :param data: a dataframe to remove columns from
    :return: the dataframe after filtering
    """
    return data.loc[:, data.any()]


def split_data(df, drop):
    """
    Splits the data to a train and test set
    :param df: the dataframe to split
    :param drop: the column name to drop
    :return: train set, train label, test set, test label
    """
    rf_target = pd.DataFrame(df[drop])
    X = df.drop("SampleID", axis=1, errors="ignore").drop(drop, axis=1, errors="ignore")
    # drops id column and default name from df
    X_train, X_test, y_train, y_test = train_test_split(
        X, rf_target.values.ravel(), test_size=0.2
    )
    # splitting the df to train and test values
    return X_train, X_test, y_train, y_test


def split_array(arr, test_size=0.2, seed=None):
    """
    Splits an array to a train and test list
    :param arr: the array to split
    :param test_size: the test size to split into (default: 0.2)
    :param seed: the shuffle seed (optional)
    :return: the split array (train data, test data)
    """
    random.shuffle(arr) if seed is None else random.Random(seed).shuffle(arr)
    return arr[: int(len(arr) * (1 - test_size))], arr[int(0 - len(arr) * test_size) :]
    # returns a split of a df


def merge_dfs(df1, df2, merged_column="SampleID"):
    """
    Merges 2 dataframes on 1 column
    :param df1: first dataframe
    :param df2: second dataframe
    :param merged_column: the column to merge on (default: 'SampleID')
    :return: the 2 dataframes merged
    """
    return df1.merge(df2, on=merged_column)


def relative_abundance(data):
    """
    Calculates the relative abundance of values in a dataframe
    :param data: the dataframe to calculate the relative abundance on
    :return: the filtered dataframe
    """
    relative_data = data
    columns_list = relative_data.columns.values.tolist()
    columns_list.remove("SampleID")
    # list of data column names (except sample ids)
    columns_sums = relative_data[columns_list].sum()
    # list of column value sums
    for row in range(len(columns_list)):
        relative_data[columns_list[row]] /= columns_sums[row]
    # dividing each element in all columns by the sum of the column

    return relative_data


def get_label_indexes(df, label, column_name="PatientGroup"):
    """
    Returns the indexes that have the specified label
    :param df: the dataframe to check
    :param label: the label to return indexes of
    :param column_name: the column name that contains the labels (default: 'PatientGroup')
    :return: the filtered dataframe
    """
    return list(df[df[column_name] == int(label)].index.values)


def encode(y):
    le = LabelEncoder()
    return le.fit_transform(y)


def filter_data(abundance, meta, control, labels=list(), label1="", label2=""):
    """
    Filters DataFrame by labels, and assigns a numeric value for each label
    :param abundance: A DataFrame that stores the features
    :param meta: A DataFrame that stores the target variable
    :param control: The control group label (will be set to 0)
    :param labels: A label list that are positive and will be set to 1 (optional)
    :param label1: A label that will be set to 1 (optional)
    :param label2:  A label that will be set to 2 (optional)
    :return: A new abundance DataFrame with the filtered samples and a new meta DataFrame with the filtered labels
    """
    df_filtered_meta = meta.loc[
        meta["PatientGroup"].isin(labels + [control] + [label1] + [label2])
    ]
    # filtering meta by parameters
    df_filtered_abundance = abundance.loc[
        abundance["SampleID"].isin(df_filtered_meta["SampleID"])
    ]
    # filtering abundance by columns that are in filtered meta (SampleID)
    df_new_filtered_meta = df_filtered_meta.copy()
    if len(labels) != 0:
        df_new_filtered_meta["PatientGroup"] = df_new_filtered_meta[
            "PatientGroup"
        ].apply(lambda x: 0 if x == control else 1)
        # mapping falsy values to 0 and truthy values to 1
    else:
        df_new_filtered_meta["PatientGroup"] = df_new_filtered_meta[
            "PatientGroup"
        ].apply(lambda x: 0 if x == control else 1 if x == label1 else 2)
        # mapping values to 0/1/2 based on labels
    df_new_filtered_meta["PatientGroup"] = df_new_filtered_meta["PatientGroup"].astype(
        int
    )
    # ensure the column is of integer type
    return df_filtered_abundance, df_new_filtered_meta


def remove_rare_features(df, prevalence_cutoff=0.1, avg_abundance_cutoff=0.005):
    """
    Remove rare features from a DataFrame
    :param df: The input DataFrame
    :param prevalence_cutoff: All features that are less prevalent than this cutoff will be removed (optional, default: 0.1)
    :param avg_abundance_cutoff: All features that have a lower average abundance than this cutoff will be removed (optional, default: 0.005)
    :return: The DataFrame with features removed
    """
    filt_df = df.drop("SampleID", axis=1) if "SampleID" in df.columns else df
    n_samples = df.shape[0]

    # Prevalence calculations (number of non-zero values per feature)
    frequencies = (filt_df > 0).sum(axis=0) / n_samples
    filt_df = filt_df.loc[:, frequencies > prevalence_cutoff]

    # Average abundance calculations
    avg_abundances = filt_df.sum(axis=0) / n_samples
    filt_df = filt_df.loc[:, avg_abundances > avg_abundance_cutoff]

    filt_df["SampleID"] = df["SampleID"]
    return filt_df


def remove_string_columns(df):
    """
    Remove the column 'SampleID' from the DataFrame
    :param df: The input DataFrame
    :return: The input DataFrame with the column 'SampleID' removed
    """
    return df.drop("SampleID", axis=1, errors="ignore")


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

    if len(set(y)) == 2:
        fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:, 1])

    else:
        y_onehot_test = LabelBinarizer().fit_transform(y)
        fpr, tpr, thresholds = roc_curve(
            y_onehot_test.ravel(), model.predict_proba(X).ravel()
        )

    return fpr, tpr, thresholds
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
    if len(set(y)) == 2:
        precision, recall, thresholds = precision_recall_curve(
            y, model.predict_proba(X)[:, 1]
        )
    else:
        y_onehot_test = LabelBinarizer().fit_transform(y)
        precision, recall, thresholds = precision_recall_curve(
            y_onehot_test.ravel(), model.predict_proba(X).ravel()
        )
    return precision, recall, thresholds


def get_explainers(model, X_test, feature_names):
    feature_importances = pd.DataFrame(
        get_feature_importance(model).T,
        index=feature_names,
        columns=["Importance"],
    )
    # feature importance
    shap_values = shap.explainers.Permutation(
        model.predict, X_test, max_evals=1000
    ).shap_values(X_test)
    # shap values
    return feature_importances, shap_values


def import_moco(path):
    modelcomparison = ModelComparison()
    for idx in range(len(os.listdir(os.path.join(path, "splits")))):
        general_split_dir = os.path.join(path, "splits", f"split_{idx}")
        train_indices_path = os.path.join(general_split_dir, "train.csv")
        test_indices_path = os.path.join(general_split_dir, "test.csv")

        train = np.loadtxt(train_indices_path, delimiter=",", dtype=int)
        test = np.loadtxt(test_indices_path, delimiter=",", dtype=int)
        modelcomparison.__splits.append((train, test))

    model_stats = []
    for model_idx in range(len(os.listdir(path)) - 1):
        model_dir = os.path.join(path, f"model_{model_idx}")
        model_pkl_path = os.path.join(model_dir, "model.pkl")

        with open(model_pkl_path, "rb") as model_pkl_file:
            model = clone(pickle.load(model_pkl_file))

        model_stats.append(ModelStats(model))

        if os.path.exists(os.path.join(model_dir, "fit_models")):
            fit_models_dir = os.path.join(model_dir, "fit_models")
            for split_index in range(len(modelcomparison.__splits)):
                fit_model_pkl_path = os.path.join(
                    fit_models_dir, f"split_{split_index}.pkl"
                )
                with open(fit_model_pkl_path, "rb") as fit_model_pkl_file:
                    fit_model = pickle.load(fit_model_pkl_file)
                model_stats[model_idx].append_fit_model(fit_model)

        for metric in METRICS.keys():
            if os.path.exists(os.path.join(model_dir, metric)):
                metric_dir = os.path.join(model_dir, metric)
                metric_instance = Metric(name=metric)
                for split_index in range(len(modelcomparison.__splits)):
                    metric_csv_path = os.path.join(
                        metric_dir, f"split_{split_index}.csv"
                    )
                    metric_values = np.loadtxt(metric_csv_path, delimiter=",")
                    metric_instance.append(metric_values)
                    setattr(model_stats[model_idx], metric, metric_instance)
        if os.path.exists(os.path.join(model_dir, "y_true")):
            for split_index in range(len(modelcomparison.__splits)):
                y_true_csv_path = os.path.join(
                    model_dir, "y_true", f"split_{split_index}.csv"
                )
                y_true_values = np.loadtxt(y_true_csv_path, delimiter=",")
                model_stats[model_idx].precision_recall_curve.y_true.append(
                    y_true_values
                )
                y_pred_csv_path = os.path.join(
                    model_dir, "y_pred", f"split_{split_index}.csv"
                )
                y_pred_values = np.loadtxt(y_pred_csv_path, delimiter=",")
                model_stats[model_idx].precision_recall_curve.y_pred.append(
                    y_pred_values
                )
    modelcomparison.__model_stats = model_stats
    return modelcomparison
