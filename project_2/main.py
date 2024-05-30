import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('diabetes.csv')

dependent_Y = df['Outcome']
independent_X = df.drop(['Outcome'], axis = 'columns')

#split the data between training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(independent_X, dependent_Y, train_size = 0.8, random_state = 6)


#LinearRegression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

Y_predicted = lr.predict(X_test)

#predicted values should be either 0 or 1 thats why the following condition we will use to get the Binary outcome
for idx in range(len(Y_predicted)):
    if Y_predicted[idx] < 0.5:
        Y_predicted[idx] = 0
    elif Y_predicted[idx] > 0.5:
        Y_predicted[idx] = 1

print("The first five predicted values are: ")
print(Y_predicted[0:5])
print("The first five predicted values are:")
print(np.array(Y_test[0:5]))

#coef and intercept
print(lr.coef_)
print(lr.intercept_)


#K-NearestNeighbour
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_predicted = knn.predict(X_test)
print("The first five predicted values are: ")
print(Y_predicted[0:5])
print("The first five predicted values are:")
print(np.array(Y_test[0:5]))
knn.score(X_test, Y_test)

#increasing the K value
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, Y_train)
Y_predicted = knn.predict(X_test)
print("The first five predicted values are: ")
print(Y_predicted[0:5])
print("The first five predicted values are:")
print(np.array(Y_test[0:5]))
knn.score(X_test, Y_test)

#increasing the K value again
knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_train, Y_train)
Y_predicted = knn.predict(X_test)
print("The first five predicted values are: ")
print(Y_predicted[0:5])
print("The first five predicted values are:")
print(np.array(Y_test[0:5]))
knn.score(X_test, Y_test)


#visual representation
from sklearn import metrics
matrix = metrics.confusion_matrix(Y_predicted, Y_test)
sns.heatmap(matrix, annot = True, cmap = "Reds", fmt = '0.2f')
plt.xlabel("Predicted Values")
plt.ylabel("Test Values")
plt.title("Confusion Matrix")
plt.show()