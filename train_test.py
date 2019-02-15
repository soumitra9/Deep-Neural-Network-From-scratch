import os
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from math import log, exp

#Initializing Data
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
df = pd.read_csv("/home/soumitra/Downloads/Iris.csv")

#split_data
X_train, X_test, y_train, y_test = train_test_split(df[[u'SepalLengthCm', u'SepalWidthCm', u'PetalLengthCm',\
                                                        u'PetalWidthCm']], df['Species'], test_size=0.2)
#Hot encode Y_train
values = np.array(y_train)


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)


# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
input_target_encoded = onehot_encoder.fit_transform(integer_encoded)

X_train = X_train.values
X_test = X_test.values
Y_train = input_target_encoded

