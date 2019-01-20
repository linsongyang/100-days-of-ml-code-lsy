# given a set of data, find the linear regression line of best fit
# and predict results
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# show plot as soon as created
# from matplotlib import interactive
# interactive(True)

scriptDir = os.path.dirname(os.path.realpath(__file__))
csvFilePath = os.path.normpath(os.path.join(scriptDir, "../datasets/studentscores.csv"))
dataset = pd.read_csv(csvFilePath)
X = dataset.iloc[ : ,   : 1 ].values # start from 1st column, take one column, all rows
Y = dataset.iloc[ : , 1 ].values # take column #1, which is the second column, all rows

# take 25% as test data, 75% as training data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1 )

# using training set to work out the linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# predict results
Y_predict = regressor.predict(X_test)

# plot the results
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color="blue")

plt.scatter(X_test, Y_test, color = "cyan")
plt.plot(X_test, regressor.predict(X_test), color = "yellow")

# show the plot graph result
plt.show()