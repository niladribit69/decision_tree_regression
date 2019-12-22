import numpy as np
import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor


# Load data
boston = datasets.load_boston()
print(boston.data.shape, boston.target.shape)
print(boston.feature_names)
data = pd.DataFrame(boston.data,columns=boston.feature_names)
data = pd.concat([data,pd.Series(boston.target,name='MEDV')],axis=1)


X = data.iloc[:,:-1]
y = data.iloc[:,-1]



x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.10,random_state=42,
shuffle=True)


# Fit regression model
# Estimate the score on the entire dataset, with no missing values
regressor =  DecisionTreeRegressor(max_depth=5,random_state=0)
regressor.fit(x_training_set, y_training_set)

y_predicted = regressor.predict(x_test_set)




##visualizing data
sns.pairplot(data)
sns.regplot(X.iloc[:,0].values,y, fit_reg= False)
sns.regplot(X.iloc[:,1].values,y, fit_reg= False)
sns.regplot(X.iloc[:,2].values,y, fit_reg= False)
sns.regplot(X.iloc[:,3].values,y, fit_reg= False)
sns.regplot(X.iloc[:,4].values,y, fit_reg= False)

sns.regplot(X.iloc[:,5].values,y, fit_reg= False)
sns.regplot(X.iloc[:,6].values,y, fit_reg= False)
sns.regplot(X.iloc[:,7].values,y, fit_reg= False)
sns.regplot(X.iloc[:,8].values,y, fit_reg= False)
sns.regplot(X.iloc[:,9].values,y, fit_reg= False)

sns.regplot(X.iloc[:,10].values,y, fit_reg= False)
sns.regplot(X.iloc[:,11].values,y, fit_reg= False)
sns.regplot(X.iloc[:,12].values,y, fit_reg= False)




'''
#visualizing decision tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(regressor, out_file=dot_data, max_depth=5,
filled=True, rounded=True,
special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
'''
