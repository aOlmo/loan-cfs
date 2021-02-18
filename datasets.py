import numpy as np
import pandas as pd

from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def get_adult(npy=False):
    full_path = 'data/UCI_adult.csv'
    df = read_csv(full_path, na_values='?')
    df = df.dropna()
    if not npy:
        return df
    else:
        X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]

        cat_cols = X.select_dtypes(include=['object', 'bool']).columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        ct = ColumnTransformer([('cat', OneHotEncoder(), cat_cols), ('num', MinMaxScaler(), num_cols)])

        X = ct.fit_transform(X).A
        y = LabelEncoder().fit_transform(y)
        return X, y


def get_credit():
    full_path = 'data/UCI_credit_approved.csv'
    column_names = ["Male", "Age", "Debt", "Married", "BankCustomer", "EducationLevel", "Ethnicity", "YearsEmployed",
                    "PriorDefault", "Employed", "CreditScore", "DriversLicense", "Citizen", "ZipCode", "Income", "Approved"]

    df = read_csv(full_path, names=column_names, header=None, dtype={"ZipCode": object})
    df = df.dropna()

    replmnts = {"Male": {'a': 1, 'b': 0}, "Approved": {'+': 1, '-': 0}}
    df = df.replace({'t': 1, 'f': 0})
    df = df.replace(replmnts)

    return df

get_credit()
