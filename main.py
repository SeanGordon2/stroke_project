import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns; sns.set()
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

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


def remove_id(df):
    """
    Drop id column from dataframe.
    :param df:
    :return: DataFrame without id column.
    """
    drop_cols = ['id']
    df_new = df.drop(columns=drop_cols)

    return df_new


def load_data_clean_and_split(dataset_name):
    """
    Function to load dataset from directory; fill null values; convert data from categorical to numerical; and split
    into training, validation, and test datasets.
    :param dataset_name: string format name of CSV file.
    :return: DataFrames containing training set, validation set, test set.
    """
    df = read_csv(dataset_name)
    df = categorical_variable_encoding(df)
    df = remove_id(df)
    train_set, val_set, test_set = func_train_test_split(df)
    train_set, val_set, test_set = fill_missing_set_data(train_set, val_set, test_set)

    x_train = train_set.drop(columns=['stroke'])
    x_val = val_set.drop(columns=['stroke'])
    x_test = test_set.drop(columns=['stroke'])

    y_train = train_set['stroke']
    y_val = val_set['stroke']
    y_test = test_set['stroke']

    return x_train, y_train, x_val, y_val, x_test, y_test


def scale_features(df):
    """
    Set up StandardScaler function to scale only numerical data specific to dataset.
    :param: df
    :return: preprocessor initialised to dataset columns.
    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    binary_features = [col for col in df.columns if sorted(df[col].dropna().unique()) == [0, 1]]

    # Remove binary features from numeric features if there's overlap
    numeric_features = [col for col in numeric_features if col not in binary_features]

    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features),
                                                   ('bin', 'passthrough', binary_features)])

    print(f"Numeric features to scale: {numeric_features}")
    print(f"Binary features passed through: {binary_features}")

    return preprocessor


def pipeline_and_random_forest(df):
    """
    Set up scaler and model to use - can edit model easily at this point.
    :return: model set up with pipeline.
    """
    preprocessor = scale_features(df)
    model = Pipeline(steps=[('preprocess', preprocessor),
                            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))])
    # model = Pipeline(steps=[('preprocess', preprocessor), ('classifier', linear_model.LogisticRegression())])
    return model


def load_data_fit_model(dataset_name):
    """
    Function to load data, clean it, split it, fit model, and evaluate on the validation set.
    :param dataset_name: string input of CSV name.
    :return: model fitted to training set data, and outputted test/validation sets.
    """
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_clean_and_split(dataset_name)

    x_base = x_train[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]  # Baseline model
    x_val_base = x_val[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]
    x_test_base = x_test[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]

    model = pipeline_and_random_forest(x_base)
    fitted_model = model.fit(x_base, y_train)
    score = fitted_model.score(x_val_base, y_val)
    print(f"Test accuracy: {score: .3f}")

    return fitted_model, x_val_base, y_val, x_test_base, y_test


def evaluation_function(model, x, y):
    """
    Evaluate fitted model performance.
    :param model: Fitted model.
    :param x: Validation/test set feature set.
    :param y: Validation/test set outcomes.
    :return: Accuracies
    """
    # Predict classes and probabilities
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)[:, 1]

    # Accuracy
    acc = accuracy_score(y, y_pred)

    # ROC AUC
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    print(f"Accuracy: {acc: .3f}")
    print(f"ROC AUC: {auc: .3f}")
    print(f"Precision: {precision: .3f}")
    print(f"Recall: {recall: .3f}")
    print(f"F1-score: {f1: .3f}")
    print("Confusion matrix:")
    print(cm)

    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc: .3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


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
    """
    Scaling features, ensuring to fit scaler after splitting data to ensure no data leakage, then transforming the
    test set afterwards.
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_test)

    model = logistic_regression_model(x_train_scaled, y_train)
    apply_ml_model(model, x_val_scaled, y_test)

    return "Finished job."


def prelim_data_analysis(x, y):
    """
    Function to analyse data set to assess distributions, etc. Uncomment sections to view graphs.
    :param x: DataFrame with cleaned data.
    :param y: DataFrame with outcomes.
    :return:
    """
    df = x.copy()
    df['stroke'] = y
    print(df.columns)
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


def feature_importance(x, y):
    """
    Preliminary feature importance selection.
    :param x:
    :param y:
    :return:
    """
    model = RandomForestClassifier()
    model.fit(x, y)

    importance = pd.Series(model.feature_importances_, index=x.columns)
    importance.sort_values(ascending=False).plot(kind='barh')
    plt.show()


def apply_scaler(x_train, y_train, x_test, y_test):
    """
    Scaling features, ensuring to fit scaler after splitting data to ensure no data leakage, then transforming the
    test set afterwards.
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_test)


def model_iteration(df):
    """

    :param df:
    :return:
    """
    train_set, val_set, test_set = load_data_clean_and_split(df)
    x_train = train_set.loc[:, train_set.columns != "stroke"]
    y_train = train_set["stroke"]
    x_val = val_set.loc[:, val_set.columns != "stroke"]
    y_val = val_set["stroke"]
    apply_test_logr(x_train, y_train, x_val, y_val)


def main_func(dataset_name):
    # x_train, y_train, x_val, y_val, x_test, y_test = load_data_clean_and_split(dataset_name)
    # prelim_data_analysis(x_train, y_train)
    # feature_importance(x_train, y_train)

    fitted_model, x_val_base, y_val, x_test_base, y_test = load_data_fit_model(dataset_name)
    evaluation_function(fitted_model, x_val_base, y_val)

    return 0


if __name__ == '__main__':
    # print(read_csv("healthcare-dataset-stroke-data.csv").head())
    # print(list(read_csv("healthcare-dataset-stroke-data.csv")))
    print(main_func("healthcare-dataset-stroke-data.csv"))
