
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def maleFemale_col_enocder(A:np.array) -> np.array:
    o = np.zeros(A.shape[0], dtype=np.int8)
    #male= 0, female=1
    for i in range(0, A.shape[0]):

        if A[i] == 'female':
            o[i] = 1
        else:
            o[i] = 0

    return o

'''
for encoded, original in zip(maleFemale_col_enocder(data["Sex"].values), data["Sex"].values):

    print(f"encoded: {encoded}, original: {original}")

'''

paths = ["titanic_data/train.csv", "titanic_data/test.csv"]


data = pd.read_csv(paths[0])
data["Sex"] = maleFemale_col_enocder(data["Sex"].values)
print(data.describe())
print(data.info())
print(data.isnull().sum())
print(data.head(10))



ids_names = data[["PassengerId", "Name"]]
data = data.drop(["Name"], axis=1)


'''
Port of Embarkation 	
C = Cherbourg, 
Q = Queenstown, 
S = Southampton
'''
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
            o[i] = 0

    return o

data["Embarked"] = encode_port(data["Embarked"])

def decode_port(A:np.array) -> np.array:
    o = np.full(A.shape[0], "", dtype=str)

    for i in range(0, A.shape[0]):
        if A[i] == 1:
            o[i] = 'C'

        elif A[i] == 2:
            o[i] = 'Q'

        elif A[i] == 3:
            o[i] = 'S'
        else:
            o[i] = ""

    return o
"""
for original, encoded, decoded in zip(data["Embarked"].values, encode_port(data["Embarked"].values), decode_port(encode_port(data["Embarked"].values))):
    print(f"encoded: {encoded}, original: {original}, decoded: {decoded}")

print(encode_port(data["Embarked"].values))
"""

print(data.columns)
#for now just dropping cabin and ticketno
data = data.drop(["Cabin", "Ticket"], axis=1)

def clean_age(A:np.array) -> np.array:
    o = np.zeros(A.shape[0], dtype=np.int32)

    for i in range(0, A.shape[0]):
        if np.isnan(A[i]):
            #apply imputations here
            o[i] = 0

        elif A[i] == None:
            o[i] = 0

        else:
            o[i] = A[i]
    return o

#print(clean_age(data["Age"].values))

data["Age"] = clean_age(data["Age"].values)
data = data.drop("PassengerId", axis=1)
print(data.head())
sns.pairplot(data=data, hue="Survived", diag_kind="hist")
plt.show()

X = data.drop("Survived", axis=1).values
y = data["Survived"].values
print(X.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scores = [0.0, 0.0]
for i in range(2, 150):
    rndForrest = RandomForestClassifier(n_estimators=i, random_state=42)
    rndForrest.fit(x_train, y_train)
    score = rndForrest.score(x_test, y_test)
    scores.append(score)
    print(f"i: {i} , score: {score}, n_estimators: {rndForrest.n_estimators}")

scores = np.array(scores)
maxIndex = np.argmax(scores)
print(f"bestScore: {scores[maxIndex]}, n_estimators: {maxIndex}")

import numpy as np
from sklearn.metrics import confusion_matrix

"""
    Input: array (n) shape with probabilites 
"""
def predictProbs(A:np.array, threshold=0.5) -> np.array:
    threshold = np.float64(threshold)

    j = 0
    customThreshold = np.zeros(A.shape[0], dtype=int)
    while j < A.size:
        if j >= A.shape[0]:
            break

        if A[j] >= threshold:
            # print(f"  at index: {j}, we predict churn = 1 :prob= {wantedPred[j]}")
            customThreshold[j] = 1
        else:
            customThreshold[j] = 0

        j += 1

    return customThreshold