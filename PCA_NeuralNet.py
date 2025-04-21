import pandas as pd

# Loading the dataset 
df = pd.read_csv("./titanic_data/train.csv")

# EDA
print(df.head())

# Basic info (types, missing values)
print(df.info())

# Summary stats
print(df.describe(include='all'))

# Check for missing values
print(df.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Survived', data=df)
plt.title("Survival Counts")
plt.show()

sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival by Sex")
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title("Survival by Passenger Class")
plt.show()

sns.histplot(df['Age'].dropna(), kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()

import numpy as np

# df_numeric = df.drop(['Cabin', 'Name', 'Ticket', 'Sex', 'Embarked'], axis=1)
# corr = df_numeric.corr()


# # This was really helpful
# sns.heatmap(corr, annot=True, cmap='coolwarm') 
# plt.title("Correlation Heatmap")
# plt.show()

# Drop unusable/noisy 
df = df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

# I'm imputing using the median and mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Converting categorical to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df_numeric = df.drop(['Embarked'], axis=1)
corr = df_numeric.corr()


# This was really helpful
sns.heatmap(corr, annot=True, cmap='coolwarm') 
plt.title("Correlation Heatmap")
plt.show()

# One hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# I'm performing feature engineering to see if the results would differ
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 0  # Default to "not alone"
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1  # If FamilySize == 1 -> alone
df.drop(['SibSp', 'Parch'], axis=1, inplace=True) # Dropping these off, since I'm using/
                                                  # engineering them above

# Preparing for PCA
X = df.drop('Survived', axis=1)  # Features
y = df['Survived']               # Labels

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

import matplotlib.pyplot as plt
import numpy as np

explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(explained_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()

pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X_scaled) # maxtrix of Eigen vectors * scaled_x 


# Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42) # TT-split handles all the dimensions appropriately. I LOVEE IT!

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary output (0 or 1)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
 
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Eigenvalues (variance explained by each PC)
print("Explained variance (eigenvalues):")
print(pca.explained_variance_)

# Percentage of variance explained by each PC
print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)







### TESTING ON THE TEST CSV ###

test_df = pd.read_csv("./titanic_data/test.csv")

# LINING UP THE DATA, Performing the same clean-ups as the test data
# Drop columns
test_df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

# Fill missing Age and Fare
test_df['Age'].fillna(df['Age'].median(), inplace=True)
test_df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Encode Sex
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# Embarked encoding (create dummies to match training)
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

# Feature Engineering
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['IsAlone'] = 0
test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1
test_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# Applying PCA
# Using the same scaler and PCA used during training (I do not need to change this!!)
X_test_scaled = scaler.transform(test_df.drop('PassengerId', axis=1))
X_test_pca = pca.transform(X_test_scaled)

predictions = model.predict(X_test_pca)
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to 0 or 1

output = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions.reshape(-1)
})
output.to_csv('submission.csv', index=False)





