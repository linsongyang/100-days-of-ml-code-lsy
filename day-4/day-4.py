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
from pandas import DataFrame
import os # io
from string import ascii_uppercase
from sklearn.model_selection import train_test_split # train set
from sklearn.preprocessing import StandardScaler # feature scaling
from sklearn.linear_model import LogisticRegression # log regression
from sklearn.metrics import confusion_matrix # matrix
from matplotlib import pyplot as plt
import seaborn as seaborn

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
classifier = LogisticRegression()
classifier = classifier.fit(X_train, Y_train)

# predication
Y_pred = classifier.predict(X_test)
print(Y_pred)

matrix = confusion_matrix(Y_test, Y_pred)
#indexes = ['test %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(Y_test))]]
#columns = ['pred %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(Y_pred))]]
#dataFrame = pd.DataFrame(matrix, index=indexes, columns=columns)

# visualization
from matplotlib.colors import ListedColormap
X_set,y_set=X_train,Y_train
X1,X2=np. meshgrid(np. arange(start=X_set[:,0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                   np. arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
plt. contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt. xlim(X1.min(),X1.max())
plt. ylim(X2.min(),X2.max())
for i,j in enumerate(np. unique(y_set)):
    plt. scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)

plt. title(' LOGISTIC(Training set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt. legend()
plt. show()

X_set,y_set=X_test,Y_test
X1,X2=np. meshgrid(np. arange(start=X_set[:,0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                   np. arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))

plt. contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt. xlim(X1.min(),X1.max())
plt. ylim(X2.min(),X2.max())
for i,j in enumerate(np. unique(y_set)):
    plt. scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)

plt. title(' LOGISTIC(Test set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt. legend()
plt. show()
