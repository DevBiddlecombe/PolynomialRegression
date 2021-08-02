#This program will test the relationship between different year model S1000RR's and their price.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

#Training set
x_train = [[2010], [2011] , [2012] , [2013] , [2014] , [2015] , [2016] , [2017] , [2018] , [2019] , [2020]]
y_train = [[135990] , [135500] , [139999] , [139888] , [144900] , [179999] , [175900] , [174900] , [199888] , [280000] , [320000]]

#Testing set
x_test = [[2010] , [2012] , [2014] , [2016] , [2018]]
y_test = [[136000] , [138000] , [150000] , [175000] , [205000]]

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(2010, 2022, 1000)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('BMW S1000RR Pricing on year')
plt.xlabel('Year of motorcycle')
plt.ylabel('Price in rands')
plt.axis([2008, 2022, 100000, 400000])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print (X_train)
print (X_train_quadratic)
print (X_test)
print (X_test_quadratic)
