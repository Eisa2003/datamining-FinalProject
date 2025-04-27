import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def maleFemale_col_enocder(A:np.array) -> np.array:
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
            o[i] = 0

    return o

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


