#%%
#import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as mplot

#%%
#importing my dataset
dataSet=pd.read_csv('neo.csv')
dataSet
# %%
#extracting dependent and independent variables
x=dataSet.iloc[:,:-1].values
y=dataSet.iloc[:,4].values


# %%
#display column y
y
# %%
#display columns x
x
# %%
#remove first dummy variable to avoid multicollinearity
x=x[:,1:]


# %%
#split into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=0)


# %%
#fitting the machine learning multivariate regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
# %%
#run the test result to see prediction------PREDICTION
yPrediction=regressor.predict(x_test)
# %%
yPrediction
# %%
y_train
# %%
x_train
# %%
print('Train Score: ',regressor.score(x_train,y_train))
print('Test Score: ',regressor.score(x_test,y_test))


