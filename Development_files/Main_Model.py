import PreProcessing

import pandas as pd
import numpy as np


paths = ["titanic_data/train.csv", "titanic_data/test.csv"]
data_train = pd.read_csv(paths[0])
data_test = pd.read_csv(paths[1])

from sklearn.model_selection import train_test_split


