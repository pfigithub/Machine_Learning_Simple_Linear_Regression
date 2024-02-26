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