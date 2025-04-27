import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def PreProcess_pipeline(df):
    df['Age'] = clean_age(df["Age"].values)
    df['Embarked'] = encode_port(df["Embarked"].values)
    df['Sex'] = Sex_encoder(df["Sex"].values)
    df['Fare'] = impute_fare(df["Fare"].values)
    df = engineer_features(df)

    return df

"""
    If used df has to have been preprocessed and encoded so features are complete
"""
def Scale_wanted_features(df, scaler, fit=False):
    X_scale = df.drop(["Embarked","Sex", "FamilySize", "IsAlone"], axis=1)
    if fit:
        scaler.fit(X_scale)

    X_scaled = scaler.transform(X_scale)

    X_notscaled = df[["Embarked", "Sex", "FamilySize", "IsAlone"]].values
    df = np.column_stack((X_scaled, X_notscaled))

    return df
def engineer_features(df):
    # I'm performing feature engineering to see if the results would differ
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0  # Default to "not alone"
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1  # If FamilySize == 1 -> alone
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)  # Dropping these off, since I'm using/

    return df

def clean_age(age:np.array) -> np.array:
    mean = np.nansum(age) / np.count_nonzero(~np.isnan(age))

    for i in range(age.size):

        if np.isnan(age[i]):
            age[i] = mean

    return age

def impute_fare(fare):
    mean = np.nansum(fare) / np.count_nonzero(~np.isnan(fare))
    for i in range(fare.size):
        if np.isnan(fare[i]):
            fare[i] = mean

    return fare

def Sex_encoder(A:np.array) -> np.array:
    o = np.zeros(A.shape[0], dtype=np.int8)
    #male= 0, female=1
    for i in range(0, A.shape[0]):

        if A[i] == 'female':
            o[i] = 1
        else:
            o[i] = 0

    return o

def encode_port(A:np.array) -> np.array:

    o = np.zeros(A.shape[0], dtype=np.int8)

    for i in range(0, A.shape[0]):
        if A[i] == 'C':
            o[i] = 1

        elif A[i] == 'Q':
            o[i] = 2

        elif A[i] == 'S':
            o[i] = 3
        else:
            o[i] = 3 # Defaulted to S since that's how Eisa imputed it

    return o