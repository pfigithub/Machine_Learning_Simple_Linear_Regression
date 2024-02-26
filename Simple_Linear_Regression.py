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