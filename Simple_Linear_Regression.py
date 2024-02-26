# import needed packages
import pandas as pd
import pylab as pl
import numpy as np


# getting and reading data
df = pd.read_csv("yourdata.csv")

# take a look at the dataset
df.head()


# selecting some feautures for example our data is FuelConsumption
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# separation of test and train data
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# fiting data by linear model from sklearn
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients if you want to plot 
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# plot outputs
import matplotlib.pyplot as plt
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# using r2_score from sklearn.metrics to evaluation
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_))
