import os
import pandas as pd
import random
import xgboost
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, train_test_split

import modelcomp.models

model_names = ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'k-NN']
model_names_short = ['RF', 'XGB', 'LR', 'SVM', 'k-NN']
model_names_dict = None


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
    X = df.drop('SampleID', axis=1, errors='ignore').drop(drop, axis=1, errors='ignore')
    # drops id column and default name from df
    X_train, X_test, y_train, y_test = train_test_split(X, rf_target.values.ravel(), test_size=0.2)
    # splitting the df to train and test values
    return X_train, X_test, y_train, y_test


def split_array(arr, test_size=0.2):
    """
    Splits an array to a train and test list
    :param arr: the array to split
    :param test_size: the test size to split into (default: 0.2)
    :return: the split array (train data, test data)
    """
    random.shuffle(arr)
    return arr[:int(len(arr) * (1 - test_size))], arr[int(0 - len(arr) * test_size):]
    # returns a split of a df


def get_feature_importance(model):
    """
    Gets the feature importance of a trained model
    :param model: the trained model (to get a return value needs to be random forest, xgboost, lr or svm)
    :return:
    """
    if type(model) in [RandomForestClassifier, xgboost.XGBClassifier]:
        return model.feature_importances_
    if type(model) in [LogisticRegression, svm.SVC]:
        return model.coef_


def merge_dfs(df1, df2, merged_column='SampleID'):
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
    columns_list.remove('SampleID')
    # list of data column names (except sample ids)
    columns_sums = relative_data[columns_list].sum()
    # list of column value sums
    for row in range(len(columns_list)):
        relative_data[columns_list[row]] /= columns_sums[row]
    # dividing each element in all columns by the sum of the column

    return relative_data


def get_label_indexes(df, label, column_name='PatientGroup'):
    """
    Returns the indexes that have the specified label
    :param df: the dataframe to check
    :param label: the label to return indexes of
    :param column_name: the column name that contains the labels (default: 'PatientGroup')
    :return: the filtered dataframe
    """
    return list(df[df[column_name] == int(label)].index.values)


def get_k_fold(seed=None, splits=5, repeats=5):
    """
    Returns a repeated k-fold with the specified seed
    :param seed: set seed for model randomness (default: None)
    :param splits: number of splits for the validation (default: 5)
    :param repeats: number of repeats for the validation (default: 5)
    :return: a RepeatedKFold object
    """
    return RepeatedKFold(random_state=seed, n_splits=splits, n_repeats=repeats) if seed is not None else RepeatedKFold(
        n_splits=splits, n_repeats=repeats)


def get_models(seed_all=None, seed_dict=None):
    return [modelcomp.models.random_forest.get(seed=(
        seed_all if seed_all != None else seed_dict[model_names_short[0]] if seed_dict is not None and
                                                                             model_names_short[
                                                                                 0] in seed_dict else None)),
        modelcomp.models.xgboost.get(seed=(
            seed_all if seed_all != None else seed_dict[model_names_short[1]] if seed_dict is not None and
                                                                                 model_names_short[
                                                                                     1] in seed_dict else None)),
        modelcomp.models.logistic_regression.get(seed=(
            seed_all if seed_all != None else seed_dict[model_names_short[2]] if seed_dict is not None and
                                                                                 model_names_short[
                                                                                     2] in seed_dict else None)),
        modelcomp.models.svm.get(seed=(
            seed_all if seed_all != None else seed_dict[model_names_short[3]] if seed_dict is not None and
                                                                                 model_names_short[
                                                                                     3] in seed_dict else None)),
        modelcomp.models.knn.get()
    ]


def filter_data(abundance, meta, control, disease_list=list(), disease1='', disease2=''):
    df_filtered_meta = meta.loc[meta['PatientGroup'].isin(disease_list + [control] + [disease1] + [disease2])]
    # filtering meta by parameters
    df_filtered_abundance = abundance.loc[abundance['SampleID'].isin(df_filtered_meta['SampleID'])]
    # filtering abundance by columns that are in filtered meta (SampleID)
    df_new_filtered_meta = df_filtered_meta.copy()
    if len(disease_list) != 0:
        df_new_filtered_meta['PatientGroup'] = df_new_filtered_meta['PatientGroup'].apply(
            lambda x: 0 if x == control else 1)
        # mapping falsy values to 0 and truthy values to 1
    else:
        df_new_filtered_meta['PatientGroup'] = df_new_filtered_meta['PatientGroup'].apply(
            lambda x: 0 if x == control else 1 if x == disease1 else 2)
        # mapping values to 0/1/2 based on diseases
    df_new_filtered_meta['PatientGroup'] = df_new_filtered_meta['PatientGroup'].astype(int)
    # ensure the column is of integer type
    return df_filtered_abundance, df_new_filtered_meta


def remove_rare_species(df, prevalence_cutoff=0.1, avg_abundance_cutoff=0.005):
    filt_df = df.drop('SampleID', axis=1) if 'SampleID' in df.columns else df
    n_samples = df.shape[0]

    # Prevalence calculations (number of non-zero values per feature)
    frequencies = (filt_df > 0).sum(axis=0) / n_samples
    filt_df = filt_df.loc[:, frequencies > prevalence_cutoff]

    # Average abundance calculations
    avg_abundances = filt_df.sum(axis=0) / n_samples
    filt_df = filt_df.loc[:, avg_abundances > avg_abundance_cutoff]

    filt_df['SampleID'] = df['SampleID']
    return filt_df


def remove_string_columns(df):
    return df.drop('SampleID', axis=1, errors='ignore')


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def join_save(add=None):
    save_to = 'export'
    return os.path.join(save_to, add) if add is not None else save_to


def data_to_filename(test_class, model_name, trained_on=None):
    test_class = 'A' if type(test_class) is list else test_class
    trained_on = 'A' if type(trained_on) is list else trained_on
    # mapping test and train classes to all disease if they are more than 1 disease
    return os.path.join(test_class,
                        trained_on if trained_on is not None else test_class,
                        model_name)


def filename_to_data(filename):
    cwd = os.getcwd()
    os.chdir(filename)
    os.chdir('..')

    model_name = os.path.basename(os.getcwd())
    # getting the model name by directory
    os.chdir('..')

    trained_on = os.path.basename(os.getcwd())
    # getting the train class by directory
    os.chdir('..')

    tested_on = os.path.basename(os.getcwd())
    # getting the test class by directory

    os.chdir(cwd)
    return model_name, str(trained_on).lower(), str(tested_on).lower()
