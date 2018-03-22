import pandas as pd
import numpy as np

df = pd.read_csv('data.csv', header=None)
df.columns = df.loc[0, :]
df = df.iloc[1:, :]


M1 = np.arange(6).reshape(2,3)
M2 = np.arange(12).reshape(4,3)

Z = np.sqrt((M2 - M1[:, None])**2)


print("M1: ", M1)
print("---------")
print("M2: ", M2)
print("Z: ", Z)



