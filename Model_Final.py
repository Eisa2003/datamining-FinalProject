"""
    Authors:
        Scheer, Daniel
        Chaudhary, Eisa

    Final Project B422 Data Mining
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import PreProcessing
#CONSTS
rng_seed = 42
data_path = "titanic_data/"

df_train = pd.read_csv(data_path + "train.csv")

print(df_train.head())
print(df_train.info())                  # Basic info (types, missing values)
print(df_train.describe(include='all')) # Summary stats
print(df_train.isnull().sum())          # Check for missing values

df_train = df_train.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)
df_train = PreProcessing.PreProcess_pipeline(df_train)

X = df_train.drop('Survived', axis=1)  # Features
y = df_train['Survived'].values               # Labels

scaler = StandardScaler()
X_transformed = PreProcessing.Scale_wanted_features(X, scaler, fit=True)


#Hyper parameters
n_estimators = 335
max_features = 6
max_depth = 5

rnd_forrest_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, random_state=rng_seed)
rnd_forrest_model.fit(X_transformed, y)



# ---- Predicting for Kaggle submission ---- #
df_test = pd.read_csv(data_path + "test.csv")
data_cache = df_test.copy()

# Drop unused features
df_test.drop(['Cabin', 'Ticket', 'Name', "PassengerId"], axis=1, inplace=True)

# perform basic preprocessing.
df_test = PreProcessing.PreProcess_pipeline(df_test)
df_test = PreProcessing.Scale_wanted_features(df_test, scaler, fit=True)

# make predictions
predictions = rnd_forrest_model.predict(df_test)

# Reassemble predictions with corresponding PassnegerId for submission
result = pd.DataFrame({'PassengerId': data_cache['PassengerId'], 'Survived': predictions})

# Store results as a csv
result.to_csv(f'RNDF_NOPCA_{rnd_forrest_model.n_estimators}_{rnd_forrest_model.max_features}_{rnd_forrest_model.max_depth}.csv', index=False)

