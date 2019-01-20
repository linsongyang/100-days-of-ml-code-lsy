# log regression is also known as classification to predict which group
# the current object under observation belongs to
# the result is discrete 0 or 1, example would be if
# some one will come and vote.
# the regression measures the relationship between the
# variables and the features, by estimating the
# probabilities using the underlying log function
# and then, the probabilities are transformed into
# binary values so as to make a predication.
# the log function is actually a sigmoid function
# the values are transformed into 0 or 1 using a threshold
# the difference between log function and linear is that
# log gives binary result and linear a continuous

import numpy as np # numpy
import pandas as pd # pandas dataset manipulation
import os # io
from sklearn.model_selection import train_test_split # train set
from sklearn.preprocessing import StandardScaler # feature scaling
from sklearn.linear_model import LogisticRegression # log regression
from sklearn.metrics import confusion_matrix # matrix

# load the csv file to dataset
# contains columns: User ID,Gender,Age,EstimatedSalary,Purchased
# where the second column "Gender" is categorical
scriptDir = os.path.dirname(os.path.realpath(__file__))
csvFilePath = os.path.normpath(os.path.join(scriptDir, "../datasets/Social_Network_Ads.csv"))
dataset = pd.read_csv(csvFilePath)

X = dataset.iloc[:, 2:4].values # take columns (3, 4, 5) (Age, Salary, Purchased)
Y = dataset.iloc[:, 4].values # take last column 5, Purchased for predication

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# apply log regression
regressor = LogisticRegression()
regressor = regressor.fit(X_train, Y_train)

# predication
Y_pred = regressor.predict(X_test)
print(Y_pred)

matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
