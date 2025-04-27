import FinalProject.PreProcessing as PreProcessing

import random
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

rng_seed = 42
random.seed(rng_seed)

paths = ["../titanic_data/train.csv", "../titanic_data/test.csv"]
data_train = pd.read_csv(paths[0])
data_test = pd.read_csv(paths[1])

# PreProcess training data
data_train["Age"] = PreProcessing.clean_age(data_train["Age"].values)
data_train["Age"] = PreProcessing.normalizeCol(data_train["Age"].values)
data_train["Sex"] = PreProcessing.maleFemale_col_enocder(data_train["Sex"].values)
data_train["Embarked"] = PreProcessing.encode_port(data_train["Embarked"].values)
data_train = data_train.drop(["PassengerId", "Name", "Cabin", "Ticket", "Fare"], axis=1)

# PreProcess submission data
data_test["Age"] = PreProcessing.clean_age(data_test["Age"].values)
data_test["Age"] = PreProcessing.normalizeCol(data_test["Age"].values)
data_test["Sex"] = PreProcessing.maleFemale_col_enocder(data_test["Sex"].values)
data_test["Embarked"] = PreProcessing.encode_port(data_test["Embarked"].values)
data_cache = data_test
data_test = data_test.drop(["PassengerId", "Name", "Cabin", "Ticket", "Fare"], axis=1)



#print(data_train.head())

y_train_full = data_train["Survived"].values
x_train_full = data_train.drop(["Survived"], axis=1).values

rnd_forrest_model = RandomForestClassifier(n_estimators=175,max_features="log2", max_depth=5, random_state=rng_seed)
rnd_forrest_model.fit(x_train_full, y_train_full)

outputs = rnd_forrest_model.predict(data_test.values)

#print(outputs)

predictions = outputs

result = pd.DataFrame({'PassengerId': data_cache['PassengerId'], 'Survived': predictions})
result.to_csv(f'RNDF_{rnd_forrest_model.n_estimators}_{rnd_forrest_model.max_features}_{rnd_forrest_model.max_depth}.csv', index=False)