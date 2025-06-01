import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import zipfile
import kaggle  # Importing dataset from kaggle.
import opendatasets as od

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def read_csv(name):
    """
    Read downloaded CSV file.
    :param name: str name of dataset
    :return: DataFrame of dataset
    """
    df = pd.read_csv(name)
    return df


def analysis_missing_data(df):
    """
    Function to identify missing data in DataFrame.
    :param df: DataFrame
    :return:
    """
    cols_with_missing = [col for col in df.columns if df[col].isnull().any()]  # Columns with missing data.
    stroke_patients = df[df['stroke'] == 1]  # Stroke patients only.
    non_stroke_patients = df[df['stroke'] == 0]  # Non-stroke patients.

    stroke_cols_with_missing = [col for col in stroke_patients.columns if stroke_patients[col].isnull().any()]
    non_stroke_cols_with_missing = [col for col in non_stroke_patients.columns if
                                    non_stroke_patients[col].isnull().any()]

    return cols_with_missing, stroke_cols_with_missing, non_stroke_cols_with_missing


def func_train_test_split(df):
    """
    Function to split into training, validation, and test data. Stratifying split based on stroke column due to low
    stroke prevalence in dataset.
    :param df: DataFrame
    :return: DataFrames containing training set, validation set, test set.
    """
    train_set, temp_set = train_test_split(df, test_size=.4, random_state=42, shuffle=True, stratify=df.stroke)
    val_set, test_set = train_test_split(temp_set, test_size=.5, random_state=42, shuffle=True,
                                         stratify=temp_set.stroke)

    return train_set, val_set, test_set


def fill_missing_set_data(train_set, val_set, test_set):
    """
    Function to impute missing data for all sets with mean value from training set to prevent data leakage into
    validation or test sets.
    :param test_set:
    :param val_set:
    :param train_set:
    :return: DataFrames for filled sets.
    """
    train_copy = train_set.copy()
    val_copy = val_set.copy()
    test_copy = test_set.copy()

    # Fill missing data with mean value of valid rows.
    for col in train_set.columns:
        train_copy[col] = train_copy[col].fillna(train_set[col].mean())  # Fill training set
        val_copy[col] = val_copy[col].fillna(train_set[col].mean())  # Fill validation set
        test_copy[col] = test_copy[col].fillna(train_set[col].mean())  # Fill test set

    return train_copy, val_copy, test_copy


def categorical_variable_encoding(df):
    """
    Function to encode categorical variables to numerical data.
    :param df: DataFrame
    :return: DataFrame with encoded variables
    """
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    # print("Categorical variables: ", object_cols)

    # Apply one-hot encoder to each column with categorical data.
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = one_hot_encoder.fit_transform(df[object_cols])
    one_hot_cols = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out(object_cols))

    one_hot_cols.index = df.index  # One-hot encoding removed index; put it back
    numerical_df = df.drop(object_cols, axis=1)  # Remove categorical columns.
    one_hot_df = pd.concat([numerical_df, one_hot_cols], axis=1)  # Add one-hot encoded columns to numerical features.
    one_hot_df.columns = one_hot_df.columns.astype(str)  # Ensure all columns have string type

    return one_hot_df


def load_data_clean_and_split(dataset_name):
    """
    Function to load dataset from directory; fill null values; convert data from categorical to numerical; and split
    into training, validation, and test datasets.
    :param dataset_name: string format name of CSV file.
    :return: DataFrames containing training set, validation set, test set.
    """
    df = read_csv(dataset_name)
    df = categorical_variable_encoding(df)
    train_set, val_set, test_set = func_train_test_split(df)
    train_set, val_set, test_set = fill_missing_set_data(train_set, val_set, test_set)

    return train_set, val_set, test_set

def main_func(dataset_name):

    train_set, val_set, test_set = load_data_clean_and_split(dataset_name)

    return 0


if __name__ == '__main__':
    # print(read_csv("healthcare-dataset-stroke-data.csv").head())
    # print(list(read_csv("healthcare-dataset-stroke-data.csv")))
    print(main_func("healthcare-dataset-stroke-data.csv"))
