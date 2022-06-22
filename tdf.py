#%%
import pandas as pd

df=pd.read_csv("salary.csv")
df.head()


#Run above before running next code

# %%
#inputs
inputs = df.drop('salary_above_100k',axis='columns')
target = df['salary_above_100k']

# %%
inputs
# %%
target 
# %%
#change labels from word to numbers using sklearn
from sklearn.preprocessing import LabelEncoder
l_company= LabelEncoder()
l_job= LabelEncoder()
l_degree= LabelEncoder()

inputs['company_n']=l_company.fit_transform(inputs['company'])
inputs['job_n']=l_company.fit_transform(inputs['job'])
inputs['degree_n']=l_company.fit_transform(inputs['degree'])

inputs.head()

# %%
##droping label columns

inputs_n= inputs.drop(['company','job','degree'], axis='columns')
inputs_n

# %%
#Training your classifier
#Importing the decision tree using scikit-learn

from sklearn import tree
#make a model
model = tree.DecisionTreeClassifier()

#train out decision tree model
#using converter labels in inputs_n against targets henced suprvised learning
model.fit(inputs_n, target)
# %%
#predicting a score
model.score(inputs_n,target)
#It is one because same dataset and same train set was used

# %%
#predictions of someone working in google as master degree and sales executive
model.predict([[2,2,1]])
