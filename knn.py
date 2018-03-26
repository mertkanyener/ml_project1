import pandas as pd
import numpy as np


df = pd.read_csv('data.csv', header=None)
df.columns = df.loc[0, :]
df = df.iloc[1:, :]

X = df.loc[:, ('height', 'midtermGrade', 'hoursOfStudy')]
X['height'] = X['height'].astype(int)
X['midtermGrade'] = X['midtermGrade'].astype(int)
X['hoursOfStudy'] = X['hoursOfStudy'].astype(int)
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

    train = train.values
    test = test.values
    dist_M = np.sqrt(np.sum(((train - test[:, None])**2), axis=2))
    return dist_M


def predict(dist_M, k, y_train):

    y_predict = []
    for i in range(dist_M.shape[0]):

        k_nearest_idx = dist_M[i].argsort()[:k]
        print(k_nearest_idx)
        vote_dict = {'pass': 0, 'fail': 0}

        for j in k_nearest_idx:

            if y_train.iloc[j] == 'pass':
                vote_dict['pass'] += 1
            else:
                vote_dict['fail'] += 1


        if vote_dict['fail'] > vote_dict['pass']:
            y_predict.append('fail')
        else:
            y_predict.append('pass')

    return y_predict



"""
M1 = np.array([(1, 3, 5), (0, 8, 2) ,(4, 1, 7), (9, 3, 6)])

M2 = np.array([(0, 5, 2),
               (4, 3, 7)])

M_res = euclidean_dist(M1, M2)
print(M1)
print(M2)
print(M_res)

#print(M_res[0][0])

#print("min value: ", np.min(M_res[0,:]))

print(np.sqrt(14))
print(np.sqrt(57))
"""

X_train, X_test, y_train, y_test = split_data(X, y)

dist_M = euclidean_dist(X_train, X_test)


y_predict = predict(dist_M, 3, y_train)

print(y_test)
print(y_predict)

"""
min3 = dist_M[0].argsort()[:3]

for i in min3:
    print(dist_M[0][i])

print(np.min(dist_M[0]))

#print(dist_M.shape)

print(X_train.iloc[0].name)
print(X_train.iloc[0])
print(X_train.iloc[0][0])


print(y_train.iloc[0])
"""