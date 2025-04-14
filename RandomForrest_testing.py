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


data_train["Age"] = PreProcessing.clean_age(data_train["Age"].values)
#data_train["Age"] = PreProcessing.normalizeCol(data_train["Age"].values)
data_train["Sex"] = PreProcessing.maleFemale_col_enocder(data_train["Sex"].values)
data_train["Embarked"] = PreProcessing.encode_port(data_train["Embarked"].values)

data_train = data_train.drop(["PassengerId", "Name", "Cabin", "Ticket", "Fare"], axis=1)

#print(data_train.head())

y_train_full = data_train["Survived"].values
x_train_full = data_train.drop(["Survived"], axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full,test_size=.15, random_state=rng_seed, stratify=y_train_full)

print(f"Train array shapes:\n  {x_train.shape}\n  {y_train.shape}")
print(f"Train Test array shapes:\n  {x_test.shape}\n  {y_test.shape}")

max_features_options = ["sqrt", "log2"]
for max_features in max_features_options:
    depths = [2, 3, 4, 5]
    for depth in depths:
        scores = []
        max_no_trees = 1000
        for i in range(10, max_no_trees, 5):

            rnd_forrest_model = RandomForestClassifier(n_estimators=i, max_features=max_features, max_depth=depth, random_state=rng_seed)
            rnd_forrest_model.fit(x_train, y_train)
            modelScore = rnd_forrest_model.score(x_test, y_test)
            print(f"{max_features}::{depth}::{i}::modelScore:{modelScore}")
            scores.append(modelScore)

        noTrees = np.linspace(10, max_no_trees, int((max_no_trees - 10) / 5))

        print(f"\n{max_features}::{depth}::\n{scores}\n")

        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot()
        ax.plot(noTrees, scores)
        ax.set_title(f'Random Forest Classifier: max_features={max_features}, depth={depth}')
        ax.set_xlabel('Number of trees')
        ax.set_ylabel('Mean Accuracy [test data]')
        plt.savefig(f"../ScreenShotCache/RNDF_{max_features}_{depth}.png", bbox_inches='tight')

        scores.clear()