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

    #TODO: Scale data
    replmnts = {"Male": {'a': True, 'b': False}, "Approved": {'+': True, '-': False}}
    df = df.replace({'t': True, 'f': False})
    df = df.replace(replmnts)

    # df = df.drop(["CreditScore","Age", 'Married', "Debt", "ZipCode", "BankCustomer", "Citizen", "DriversLicense", "Ethnicity", "EducationLevel"], axis=1)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # df[["YearsEmployed", "Income"]] = scaler.fit_transform(df[["YearsEmployed", "Income"]])

    data = {
        "df": df,
        "dot_graph":
            """
                digraph {
                Male;Age;Debt;Married;BankCustomer;ZipCode;CreditScore;DriversLicense;Citizen;PriorDefault;EducationLevel;Ethnicity;YearsEmployed;Income;
                Male->Approved;
                Age->Married;Age->EducationLevel;Age->YearsEmployed;Age->CreditScore;Age->Approved; Age->Debt; Age->DriversLicense; Age->Income;
                Debt->Approved;
                Married->Approved;
                BankCustomer->Approved;
                ZipCode->BankCustomer; ZipCode->Approved;
                CreditScore->Approved;
                DriversLicense->Approved;
                Citizen->Approved;
                PriorDefault->Approved;
                EducationLevel->Approved; EducationLevel->PriorDefault;
                Ethnicity->Approved;
                YearsEmployed->Income;YearsEmployed->Approved;
                Income->Approved;Income->ZipCode;}
            """.replace("\n", " ")
        # "dot_graph":
        #     """
        #         digraph {
        #         Male->Approved;
        #         PriorDefault->Approved;
        #         YearsEmployed->Income;YearsEmployed->Approved;
        #         Income->Approved;Income->ZipCode;}
        #     """.replace("\n", " ")
    }


    return data

