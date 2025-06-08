import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
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
    stroke prevalence in dataset. Hard coded to have training set = 60%, validation set = 20%, and test set = 20%.
    :param df: DataFrame
    :return: DataFrames containing training set, validation set, test set.
    """
    train_set, temp_set = train_test_split(df, test_size=.4, random_state=42, shuffle=True, stratify=df.stroke)
    val_set, test_set = train_test_split(temp_set, test_size=.5, random_state=42, shuffle=True,
                                         stratify=temp_set.stroke)

    return train_set, val_set, test_set


def fill_missing_set_data(train_set, val_set, test_set):
    """
    Impute missing data for all sets with mean value from training set to prevent data leakage into
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


def logistic_regression_model(x, y):
    """
    Function to train logistic regression model.
    :param x: DataFrame of training data inputs.
    :param y: DataFrame of training data outputs.
    :return: Logistic Regression model fitted for training data.
    """
    logr_model = linear_model.LogisticRegression(max_iter=1000)
    logr_model.fit(x, y)

    return logr_model


def apply_ml_model(model, x, y):
    """

    :param model: Trained ML model.
    :param x: DataFrame of test inputs.
    :param y: DataFrame of test outputs.
    :return:
    """
    # Get probability estimates for the test set
    probabilities = model.predict_proba(x)
    # Make predictions using predict()
    predictions = model.predict(x)

    # # Compare predictions with probability estimates
    # for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    #     print(
    #         f"Sample {i + 1}: Predicted Class = {pred}, Class 0 Probability = {prob[0]:.4f}, "
    #         f"Class 1 Probability = {prob[1]:.4f}")

    print("Score of Model: ", model.score(x, y))
    confusion_matrix_full = confusion_matrix(y, predictions)

    cm_df = pd.DataFrame(confusion_matrix_full,
                         index=['1', '0'],
                         columns=['1', '0'])
    ax = sns.heatmap(cm_df, fmt='d', cmap="YlGnBu", cbar=False, annot=True)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()


def apply_test_logr(x_train, y_train, x_test, y_test):
    # 1. Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_test)

    model = logistic_regression_model(x_train_scaled, y_train)
    apply_ml_model(model, x_val_scaled, y_test)

    return "Finished job."


def prelim_data_analysis(df):
    """
    Function to analyse data set to assess distributions, etc. Uncomment sections to view graphs.
    :param df: DataFrame with cleaned data.
    :return:
    """
    mean_stroke_incidence = df.stroke.mean()
    print("Percentage with Stroke: ", mean_stroke_incidence)

    binary_columns = ['stroke', 'hypertension', 'heart_disease', 'gender_Female', 'gender_Male', 'gender_Other',
                      'ever_married_No', 'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
                      'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'Residence_type_Rural',
                      'Residence_type_Urban', 'smoking_status_Unknown', 'smoking_status_formerly smoked',
                      'smoking_status_never smoked', 'smoking_status_smokes']

    # Uncomment to view charts.

    # Histogram distributions of ages grouped by binary variables based on the occurrence of stroke.
    # for col in binary_columns[1:]:
    #     df[(df[str(col)] == 1) & (df.stroke == 0)].age.hist(bins=20, label='Non-Stroke')
    #     df[(df[str(col)] == 1) & (df.stroke == 1)].age.hist(bins=20, label='Stroke')
    #     plt.legend(prop={'size': 12})
    #     plt.title(str(col))
    #     plt.xlabel('Age (years)')
    #     plt.ylabel('Count')
    #     plt.show()

    # Stroke rate per variable.
    stroke_rate_per_variable = df.groupby("stroke").agg('mean')
    print(stroke_rate_per_variable)

    # Ratio of stroke incidence compared to the entire dataset.
    ratio_stroke_rate_binary_columns = df[binary_columns].groupby("stroke").agg('mean') / mean_stroke_incidence
    print(ratio_stroke_rate_binary_columns)

    # Create frequency tables of feature vs stroke rate.
    for col in binary_columns[1:]:  # Iterate through binary columns, skipping the stroke column.
        print(pd.crosstab(df[col], df['stroke'], normalize='index'))

    numerical_cols = [col for col in df.columns if col not in binary_columns[1:]]

    # Boxplots for numerical features.
    # for col in numerical_cols:
    #     df.groupby('stroke')[col].describe()
    #     sns.boxplot(data=df, x='stroke', y=str(col))
    #     plt.title(str(col))
    #     plt.show()



def main_func(dataset_name):
    train_set, val_set, test_set = load_data_clean_and_split(dataset_name)
    prelim_data_analysis(train_set)

    # x_train = train_set.loc[:, train_set.columns != "stroke"]
    # y_train = train_set["stroke"]
    # x_val = val_set.loc[:, val_set.columns != "stroke"]
    # y_val = val_set["stroke"]
    # apply_test_logr(x_train, y_train, x_val, y_val)

    return 0


if __name__ == '__main__':
    # print(read_csv("healthcare-dataset-stroke-data.csv").head())
    # print(list(read_csv("healthcare-dataset-stroke-data.csv")))
    print(main_func("healthcare-dataset-stroke-data.csv"))
