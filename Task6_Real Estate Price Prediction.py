
# coding: utf-8

# # Task6: Predicting Real Estate House Prices

# ## This task is provided to test your understanding of building a Linear Regression model for a provided dataset

# ### Dataset: Real_estate.csv

# ### Import the necessary libraries
# #### Hint: Also import seaborn

# In[2]:


import pandas as pd


# ### Read the csv data into a pandas dataframe and display the first 5 samples

# In[4]:


data=pd.read_csv('Real estate.csv')
data.head()


# ### Show more information about the dataset

# In[8]:


data.info()


# ### Find how many samples are there and how many columns are there in the dataset

# In[11]:


c=data.shape
c


# ### What are the features available in the dataset?

# In[17]:


data.columns


# ### Check if any features have missing data

# In[22]:


len(data)
data.count()
count_nan = len(data) - data.count()
count_nan


# In[25]:


data.isnull().values.any()


# ### Group all the features as dependent features in X

# In[27]:


X=data.iloc[:,:-1]
X


# ### Group feature(s) as independent features in y

# In[29]:


y=data.iloc[:,7]
y


# ### Split the dataset into train and test data

# In[118]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state=8)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Choose the model (Linear Regression)

# In[122]:


from sklearn.linear_model import LinearRegression
    


# ### Create an Estimator object

# ### Train the model

# In[121]:


lm = LinearRegression()
lm.fit(X_train, y_train)

Y_pred = lm.predict(X_test)


# ### Apply the model

# In[123]:


Y_pred


# ### Display the coefficients

# In[124]:


lm.coef_


# ### Find how well the trained model did with testing data

# In[125]:


p=lm.score(X_test,y_test)
p1=p*100
p1
print("Accuracy:",p1)


# ### Plot House Age Vs Price
# #### Hint: Use regplot in sns

# In[90]:


import seaborn as sns;
sd = sns.regplot(y="X2 house age",x="Y house price of unit area", data=data)
sd


# ### Plot Distance to MRT station Vs Price

# In[88]:


sd1 = sns.regplot(y="X3 distance to the nearest MRT station", x="Y house price of unit area", data=data)


# ### Plot Number of Convienience Stores Vs Price

# In[89]:


sd2 = sns.regplot(y="X4 number of convenience stores", x="Y house price of unit area", data=data)

