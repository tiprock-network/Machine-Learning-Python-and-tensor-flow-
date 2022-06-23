#%%
import imp
from turtle import color
import matplotlib as mpl
import matplotlib.pyplot as mplot
import numpy as np
import pandas as pd

#%%
#importing my data set
ds=pd.read_csv('security_settlements.csv')
ds
# %%
#define dependent and independent variables
#independent as x
x=ds.iloc[:, :-1].values
#dependent as y
y=ds.iloc[:,1].values

# %%
#split into training set and test set
#In the 30 observations we use 20 observations for training
#10 observations for testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=1/3, random_state=0)
# %%
#fitting model to data set
#import sklearn linear regression model
from sklearn.linear_model import LinearRegression

# %%
regressor=LinearRegression()
regressor.fit(x_train, y_train)

# %%
#prediction
y_prediction = regressor.predict(x_test)
x_prediction = regressor.predict(x_train)
# %%
#visualize training set results
mplot.scatter(x_train,y_train,color="blue")
mplot.plot(x_train,x_prediction, color="black")
mplot.title("Police vs Settlement")
mplot.xlabel("police")
mplot.ylabel("apartments")
mplot.show()
# %%
#visualize training test results
mplot.scatter(x_test,y_test,color="blue")
mplot.plot(x_train,x_prediction, color="black")
mplot.title("Police vs Settlement")
mplot.xlabel("police")
mplot.ylabel("apartments")
mplot.show()

# %%
