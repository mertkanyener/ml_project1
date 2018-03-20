import pandas as pd

df = pd.read_csv('data.csv', header=None)
df.columns = df.loc[0, :]
df = df.iloc[1:, :]

X1 = df.loc[:, ('hoursOfStudy')]
X2 = df.loc[:, ('height', 'midtermGrade')]

X1_S = X1.sample(frac=1)
X2_S = X2.sample(frac=1)


#print(X1_S)
#print(X2_S)

y = df.loc[:, 'label']
