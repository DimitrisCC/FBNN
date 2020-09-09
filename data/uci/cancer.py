import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data/uci/cancer_data.csv")
labels = pd.read_csv("data/uci/cancer_labels.csv")

data = data.drop(columns=['Unnamed: 0'])
labels = labels.drop(columns=['Unnamed: 0'])

data = data.loc[:, (data.max() != data.min())]

le = LabelEncoder()

labels = pd.DataFrame(le.fit_transform(labels.to_numpy().ravel()))

data = pd.concat([data, labels], axis=1)

data.to_csv('data/uci/cancer.data', sep=' ', header=False)
