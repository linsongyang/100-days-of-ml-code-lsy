import numpy as np
import pandas as pd
import os
import scipy

print(np.__version__)
print(pd.__version__)
print(scipy.__version__)
print("cwd is: " +  os.getcwd())

# load the dataset using pandas csv reader
scriptPath = os.path.dirname(os.path.realpath(__file__))
csvFilePath = os.path.normpath(os.path.join(scriptPath, "../datasets/Data.csv"))
print("load csv data from file: " + csvFilePath)
dataset = pd.read_csv(csvFilePath)
dataset.head()

X = dataset.iloc[ : , :-1].values # all rows, all columns except the last, the team scores
Y = dataset.iloc[ : , 3].values # all rows, only the 3rd column 0-based, yes no values
print("X values are:")
print(X)
print("Y values are:")
print(Y)

# handle the missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # algorithm, find NaN and replace with mean
imputer = imputer.fit(X[ : , 1:3]) # work out the formula
X[ : , 1:3] = imputer.transform(X[ : , 1:3]) # apply to the dataset to get new set of data
print("X values after fix are:")
print(X)

# transform team names to a index dictionary on 1st column
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
print("Labelled X values:")
print(X)

# transform the yes no values to a index dictionary on Y array
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("Labelled Y values:")
print(Y)

# onehotencoder = OneHotEncoder(categorical_features = [0])
# categorical_features : ‘all’ or array of indices or mask, default=’all’
# Specify what features are treated as categorical.
#   ‘all’: All features are treated as categorical.
#   array of indices: Array of categorical feature indices.
#   mask: Array of length n_features and with dtype=bool.
from sklearn.preprocessing import OneHotEncoder

# choose column 0, the name of the teams, as the category
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print(X)

# split dataset 80:20 training:test
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

# feature scling
# The StandardScaler assumes your data is normally distributed within each feature and will scale them
# such that the distribution is now centred around 0, with a standard deviation of 1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)