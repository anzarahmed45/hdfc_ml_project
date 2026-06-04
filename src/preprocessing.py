import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_data(path):

    df = pd.read_csv(path)

    return df


def handle_missing_values(df):

    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include="object").columns

    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def remove_duplicates(df):
    df = df.drop_duplicates()
    
    return df


def encode_categorical_columns(df):
    encoder = LabelEncoder()
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:

        df[col] = encoder.fit_transform(df[col])

    return df


def handle_outliers(df):
    numerical_cols = df.select_dtypes(include=np.number).columns

    for col in numerical_cols:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    return df


def scale_features(df):

    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df