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

    df = read_csv(full_path, names=column_names, header=None, dtype={"ZipCode": object}, na_values='?')
    df = df.dropna()

    replmnts = {"Male": {'a': 1, 'b': 0}, "Approved": {'+': 1, '-': 0}}
    df = df.replace({'t': 1, 'f': 0})
    df = df.replace(replmnts)
    df.Approved = df.Approved.astype(int)
    df.Male = df.Male.astype(int)

    # Dropping ZipCode for dimensionality reduction
    df = df.drop(["ZipCode"], axis=1)

    obj_df = df.select_dtypes(include=['object'])
    df = pd.get_dummies(df, columns=obj_df.columns)

    cols = list(df.columns.values)  # Make a list of all of the columns in the df
    cols.pop(cols.index("Approved"))  # Remove b from list
    df = df[cols + ["Approved"]]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    ct = ColumnTransformer([('num', MinMaxScaler(), num_cols)])

    df[num_cols] = ct.fit_transform(df)

    #TODO: Change all datatype to float + Scale data + Normalize over all?
    x, y = df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values

    data = {
        "df": df,
        "dot_graph":
            """
                digraph {
                Male;Age;Debt;Married;BankCustomer;CreditScore;DriversLicense;Citizen;PriorDefault;EducationLevel;Ethnicity;YearsEmployed;Income;
                Male->Approved;
                Age->Married;Age->EducationLevel;Age->YearsEmployed;Age->CreditScore;Age->Approved; Age->Debt; Age->DriversLicense; Age->Income;
                Debt->Approved;
                Married->Approved;
                BankCustomer->Approved;
                CreditScore->Approved;
                DriversLicense->Approved;
                Citizen->Approved;
                PriorDefault->Approved;
                EducationLevel->Approved; EducationLevel->PriorDefault;
                Ethnicity->Approved;
                YearsEmployed->Income;YearsEmployed->Approved;
                Income->Approved;}
            """.replace("\n", " "),
        "x_y": [x, y]
    }


    return data

