import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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


def fill_missing_data(df):
    """
    Function to impute missing data, in this case only .
    :param df: DataFrame
    :return:
    """
    df_copy = df.copy()  # Ensure original data is unchanged.
    stroke_patients_df = df_copy[df_copy['stroke'] == 1].copy()  # Stroke patients only.
    non_stroke_patients_df = df_copy[df_copy['stroke'] == 0].copy()  # Non-stroke patients.

    # Fill missing data with mean value of valid rows.
    stroke_patients_df.bmi = stroke_patients_df.bmi.fillna(stroke_patients_df.bmi.mean())
    non_stroke_patients_df.bmi = non_stroke_patients_df.bmi.fillna(non_stroke_patients_df.bmi.mean())
    # Combine the filled datasets.
    recombined_df = pd.concat([stroke_patients_df, non_stroke_patients_df], axis=0)

    return recombined_df


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


def main_func(dataset_name):
    """
    Function to run total dataset.
    :param dataset_name: string name of dataset in directory.
    :return:
    """
    df = read_csv(dataset_name)
    print([col for col in df.columns])
    full_df = fill_missing_data(df)
    cat_cols = categorical_variable_encoding(full_df)
    return cat_cols


if __name__ == '__main__':
    # print(read_csv("healthcare-dataset-stroke-data.csv").head())
    # print(list(read_csv("healthcare-dataset-stroke-data.csv")))
    print(main_func("healthcare-dataset-stroke-data.csv").head())
