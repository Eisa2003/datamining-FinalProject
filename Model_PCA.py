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

# Loading the dataset
df_train = pd.read_csv(data_path + "train.csv")

# EDA
print(df_train.head())
print(df_train.info())                  # Basic info (types, missing values)
print(df_train.describe(include='all')) # Summary stats
print(df_train.isnull().sum())          # Check for missing values

# Drop unused features
df_train = df_train.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

#PreProcess Training data
df_train = PreProcessing.PreProcess_pipeline(df_train)

# Create and plot correlation matrix
df_numeric = df_train.drop(['Embarked'], axis=1)
corr = df_numeric.corr()


# Preparing for PCA
X = df_train.drop('Survived', axis=1)  # Features
y = df_train['Survived']               # Labels

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Performing PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = np.cumsum(pca.explained_variance_ratio_)
pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X_scaled) # maxtrix of Eigen vectors * scaled_x

# Split data into train and test set to train model
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)

# Instantiate and fit the model with data from PCA
rnd_forrest_model = RandomForestClassifier(n_estimators=300, max_features=1, max_depth=4, random_state=rng_seed)
rnd_forrest_model.fit(X_reduced, y)

# Calculate fpr, tpr and thresholds from survived probabilities
proba_preds = rnd_forrest_model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, proba_preds[:, 1])
auc_roc = auc(fpr, tpr)

#Plot roc-curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for survivor Classification')
plt.legend()
plt.show()


# ---- Predicting for Kaggle submission ---- #
df_test = pd.read_csv(data_path + "test.csv")
data_cache = df_test.copy()

# Drop unused features
df_test.drop(['Cabin', 'Ticket', 'Name', "PassengerId"], axis=1, inplace=True)

# perform basic preprocessing.
df_test = PreProcessing.PreProcess_pipeline(df_test)

# Applying PCA
# Using the same scaler and PCA used during training (I do not need to change this!!)
X_test_scaled = scaler.transform(df_test)
X_test_pca = pca.transform(X_test_scaled)

# make predictions
predictions = rnd_forrest_model.predict(X_test_pca)

# Reassemble predictions with corresponding PassnegerId for submission
result = pd.DataFrame({'PassengerId': data_cache['PassengerId'], 'Survived': predictions})

# Store results as a csv
result.to_csv(f'RNDF_PCA_{rnd_forrest_model.n_estimators}_{rnd_forrest_model.max_features}_{rnd_forrest_model.max_depth}.csv', index=False)
