# import needed packages
import pandas as pd
import pylab as pl
import numpy as np


# getting and reading data
df = pd.read_csv("yourdata.csv")

# take a look at the dataset
df.head()


# selecting some feautures
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
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)