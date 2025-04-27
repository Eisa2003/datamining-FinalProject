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

print(X.head(5))

scaler = StandardScaler()

X_transformed = PreProcessing.Scale_wanted_features(X, scaler, fit=True)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.15, random_state=42, stratify=y)





max_features_options = ["log2", "sqrt", 1, 2, 4, 6]
for max_features in max_features_options:
    depths = [2, 3, 4, 5, 6]
    for depth in depths:
        scores = []
        max_no_trees = 500
        for i in range(10, max_no_trees, 5):

            rnd_forrest_model = RandomForestClassifier(n_estimators=i, max_features=max_features, max_depth=depth, random_state=rng_seed)
            rnd_forrest_model.fit(X_train, y_train)

            proba_preds = rnd_forrest_model.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, proba_preds[:, 1])
            auc_roc = auc(fpr, tpr)
            print(f"{max_features}::{depth}::{i}::modelScore:{auc_roc}")
            scores.append(auc_roc)

        noTrees = np.linspace(10, max_no_trees, int((max_no_trees - 10) / 5))

        print(f"\n{max_features}::{depth}::\n{scores}\n")

        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot()
        ax.plot(noTrees, scores)
        ax.set_title(f'Random Forest Classifier: max_features={max_features}, depth={depth}')
        ax.set_xlabel('Number of trees')
        ax.set_ylabel('AUC')
        plt.savefig(f"Screenshots/RNDF_NOPCA_{max_features}_{depth}.png", bbox_inches='tight')

        scores.clear()


