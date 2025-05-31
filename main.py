import pandas as pd
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


def main_func(dataset_name):
    """
    Function to run total dataset.
    :param dataset_name: string name of dataset in directory.
    :return:
    """
    df = read_csv(dataset_name)
    full_df = fill_missing_data(df)
    return full_df.head()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(read_csv("healthcare-dataset-stroke-data.csv").head())
    # print(list(read_csv("healthcare-dataset-stroke-data.csv")))
    print(main_func("healthcare-dataset-stroke-data.csv"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
