import pandas as pd
import numpy as np


df = pd.read_csv('data.csv', header=None)
df.columns = df.loc[0, :]
df = df.iloc[1:, :]

X = df.loc[:, ('height', 'midtermGrade', 'hoursOfStudy')]
y = df.loc[:, 'label']


def split_data(X, y, test_size=0.3):

    shuf_idx = np.random.permutation(X.index)
    X_shuffled = X.reindex(shuf_idx)
    y_shuffled = y.reindex(shuf_idx)
    length = X.count()[1]
    i = int((1.0 - test_size ) * length)
    X_train = X_shuffled.iloc[:i, :]
    X_test = X_shuffled.iloc[i:, :]
    y_train = y_shuffled.iloc[:i]
    y_test = y_shuffled.iloc[i:]

    return X_train, X_test, y_train, y_test


def euclidean_dist(train, test):

    dist_M = np.sqrt((train - test[:, None])**2)
    return dist_M





M1 = np.array([(1, 3, 5), (0, 8, 2) ,(4, 1, 7), (9, 3, 6)])

M2 = np.array([(0, 5, 2),
               (4, 3, 7)])

M_res = euclidean_dist(M1, M2)
print(M1)
print(M2)
print(M_res)

print(M_res[0][0])

print("min value: ", np.min(M_res[0,:]))
#X_train, X_test, y_train, y_test = split_data(X, y)



