import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
    random.shuffle(arr) if seed == None else random.Random(seed).shuffle(arr)
    return arr[: int(len(arr) * (1 - test_size))], arr[int(0 - len(arr) * test_size) :]
    # returns a split of a df


def get_feature_importance(model, attr=None):
    """
    Gets the feature importance of a trained model
    :param model: The trained model (to get a return value needs to have a known attribute or a custom attribute name)
    :param attr: The model's attribute name (default: None)
    :return:
    """
    if attr is not None:
        if hasattr(model, attr):
            try:
                return np.abs(getattr(model, attr))
            except AttributeError:
                try:
                    return np.abs(getattr(model, attr)())
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

    if hasattr(model, "feature_importance_"):
        return np.abs(model.feature_importances_)
    if hasattr(model, "coef_"):
        return np.abs(model.coef_)

    warnings.warn(
        f"{model.__class__.__name__} has no known attribute and no custom attribute was given, so None was returned"
    )
    return None


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


def make_dir(directory):
    """
    Creates a subdirectory, if one doesn't exist
    :param directory: The directory that needs to be added
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def join_save(add=None):
    save_to = "export"
    return os.path.join(save_to, add) if add is not None else save_to


def data_to_filename(test_class, model_name, trained_on=None):
    test_class = "A" if type(test_class) is list else test_class
    trained_on = "A" if type(trained_on) is list else trained_on
    # mapping test and train classes to all label if they are more than 1 label
    return os.path.join(
        trained_on if trained_on is not None else test_class, test_class, model_name
    )


def filename_to_data(filename):
    splits = filename.rsplit("/", 4)

    return splits[-4].lower(), splits[-3].lower(), splits[-2]
