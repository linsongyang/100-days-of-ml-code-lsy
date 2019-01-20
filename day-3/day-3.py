# multiple linear regression to examine the relations between two or more features
# terminology features used here would be the same as dimensions in data mining, I assume

import numpy as np # numerics
import pandas as pd # dataset manipulations
import os # file IO
import matplotlib.pyplot as plt # plot
from sklearn.model_selection import train_test_split # training test set split
from sklearn.linear_model import LinearRegression # linear regression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # categorical data encoding and turn categories to features

# load the csv file to dataset
# contains columns: R&D Spend,Administration,Marketing Spend,State,Profit
# where the second last column "State" is categorical; rest float
scriptDir = os.path.dirname(os.path.realpath(__file__))
csvFilePath = os.path.normpath(os.path.join(scriptDir, "../datasets/50_Startups.csv"))
dataset = pd.read_csv(csvFilePath)

# select data
X = dataset.iloc[ : , :-1].values # all rows; 1 less columns excluding the last column
Y = dataset.iloc[ : ,  4 ].values # all rows; 5th column, the Profit

# encode 4th column, the "State" column, then turn the X data to features
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3]) # pick the 4th column
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray() # turn to features, 3 different states

# avoid dummy trap, exclude the 1st feature, take from 2nd column onwards
# dummy trap is caused by highly co-related features, ie one can be predicted from the other
# for example, we can have 2 categories for male and female, but if we eliminate male, then we would
# get the result for non-female. Because the features are extracted to 2 different columns
# value 1 from male column would definitely value 0 for female column, and vice versa
# therefore, we just take -1 columns to avoid the trap.
X = X[: , 1:]

# split traning set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# linear regression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# predication
y_predict = regressor.predict(X_test)
print(y_predict)