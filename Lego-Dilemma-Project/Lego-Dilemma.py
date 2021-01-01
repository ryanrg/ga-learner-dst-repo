#!/usr/bin/env python
# coding: utf-8

# ### Load and split the dataset
# - Load the train data and using all your knowledge of pandas try to explore the different statistical properties of the dataset.
# - Separate the features and target and then split the train data into train and validation set.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('.\Pandas Test\Lego Project\Lego train.csv')


# In[3]:


#View the basic stats for all the columns in the data set
data.describe()


# In[4]:


#Count Null in Data set
pd.DataFrame(data.isnull().sum())


# In[5]:


# drop id as it has no value
data = data.drop(['Id'], axis=1)


# In[6]:


X = data.drop(['list_price'], axis = 1)
Y = data[['list_price']]


# In[7]:


X_train, X_test,y_train,y_test = train_test_split(X, Y , test_size=0.3, random_state=6)


# ### Data Visualization
# 
# - All the features including target variable are continuous. 
# - Check out the best plots for plotting between continuous features and try making some inferences from these plots. 

# In[8]:


# Code starts here

cols = X_train.columns
fig, axes = plt.subplots(nrows = 3, ncols= 3, figsize = (20, 20))

for i in range(0,3):             
    for j in range(0,3):         
            col = cols[i*3 + j]
            axes[i,j].set_title(col)
            axes[i,j].scatter(X_train[col], y_train)
            axes[i,j].set_xlabel(col) 
            axes[i,j].set_ylabel('list_price')
# Code ends here.


# ### Feature Selection
# - Try selecting suitable threshold and accordingly drop the columns.

# In[9]:


# Code starts here
corr = X_train.corr()
print(corr.round(2)) 


# In[10]:


#play_star_rating and  val_star_rating have a correlation of 0.91 its possible these are correlated.
#we drop it in the train and test data(X)
X_train = X_train.drop(['play_star_rating', 'val_star_rating'], axis=1)
X_test = X_test.drop(['play_star_rating', 'val_star_rating'], axis=1)
# Code ends here.


# In[11]:



X_test.head(5)


# ### Model building

# In[12]:


ModelReg = LinearRegression()

ModelReg.fit(X_train, y_train)
y_pred = ModelReg.predict(X_test)
#print(y_pred)

mse = round(mean_squared_error(y_test, y_pred),2)
print(f'MSE is {mse}')

r2 = round(r2_score(y_test, y_pred),2)
print(f' R^2 is {r2}') 


# ### Residual check!
# 
# - Check the distribution of the residual.

# In[13]:


# Code starts here
residual = y_test - y_pred
np.sum(residual)


# In[14]:



residual.hist()

# Code ends here.


# ### Prediction on the test data and creating the sample submission file.
# 
# - Load the test data and store the `Id` column in a separate variable.
# - Perform the same operations on the test data that you have performed on the train data.
# - Create the submission file as a `csv` file consisting of the `Id` column from the test data and your prediction as the second column.

# In[42]:


# Code starts here

data_pred = pd.read_csv('.\Pandas Test\Lego Project\Lego test.csv')

submission = pd.DataFrame(data_pred['Id'].copy())
data_pred = data_pred.drop(['Id'], axis=1)
data_pred = data_pred.drop(['play_star_rating', 'val_star_rating'], axis=1)


submission.head(4)


# In[43]:


y_pred__td = np.round(ModelReg.predict(data_pred),2)
y_pred__td_pd =  pd.DataFrame(y_pred__td)
y_pred__td_pd.head(5)


# In[45]:


submission['list_price'] = y_pred__td

submission.to_csv('.\Pandas Test\Lego Project\submission.csv', index = False)
# Code ends here.

