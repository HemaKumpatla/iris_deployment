#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION

# ### Importing required libraries and dataset

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore")


# In[5]:


df=pd.read_csv('C:/Users/hemak/OneDrive/Desktop/Iris.csv')


# ### Primary Exploration of the data

# In[6]:


df.head(5)


# In[7]:


df.tail(5)


# In[8]:


df.sample(5)


# In[9]:


def explore_data(df):
    pd.set_option('display.max_rows',100)
    pd.set_option('display.max_columns',100)

    #Get the basic information about the dataframe
    print("Data Shape:")
    print(df.shape)

    print("\nData Columns:")
    print(df.columns)
    
    print("\nData Info:\n")
    print(df.info())
    
    #Check for missing values
    print("\nMissing values:\n")
    print(df.isnull().sum())
    
    #Check for duplicate rows
    print("\nDuplicate rows\n:")
    print(df.duplicated().sum())
    
    #Explore unique values in all columns
    print("\nUnique values in all columns\n:")
    print(df.nunique())
    
explore_data(df)


# In[10]:


df.drop(['Id'],axis=1,inplace=True)


# In[11]:


pd.pivot_table(df, index='Species', values=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])


# ### Inferences:

# * No duplicates 
# * No null or missing values 
# * All input features are numerical and the target is categorical 

# ### Data Visualization

# #### [1] UNIVARIATE ANALYSIS

# **Analyzing the distribution of Species**

# In[12]:


fig=px.pie(df,names='Species',title='Distribution of Species')
fig.show()


# * Inference: The data is equally distributed. There is no imbalance.

# **Analyzing distributions of numerical columns**

# In[13]:


numerical_cols=df.select_dtypes(include=['int64','float64']).columns


# * Violin Plots

# In[14]:


for i in numerical_cols:
    fig=go.Figure(data=[go.Violin(x=df[i])])
    fig.update_layout(
    title=i,
    xaxis_title=i,
    yaxis_title="Count")
    fig.show()


# **Analyzing Target Vs Numerical columns**

# * Bar Plots

# In[15]:


for i in numerical_cols:
    plt.figure(figsize=(5,4))
    sns.barplot(x=df['Species'], y=df[i], data=df, ci=None, palette='hls')
    plt.show()


# * Box Plots

# In[16]:


for i in numerical_cols:
    plt.figure(figsize=(5,4))
    sns.boxplot(x=df['Species'], y=df[i], data=df, palette='hls')
    plt.show()


# #### [2]BIVARIATE ANALYSIS

# **Numerical Vs Numerical**

# * Sepal Length Vs Sepal Width

# In[17]:


plt.figure(figsize=(5,4))
sns.lineplot(x=df['SepalLengthCm'], y=df['SepalWidthCm'], data=df, ci=None, palette='hls')
plt.show()


# * Petal Length Vs Petal Width

# In[18]:


plt.figure(figsize=(5,4))
sns.lineplot(x=df['PetalLengthCm'], y=df['PetalWidthCm'], data=df, ci=None, palette='hls')
plt.show()


# * Sepal Length Vs Petal Width

# In[19]:


plt.figure(figsize=(5,4))
sns.lineplot(x=df['SepalLengthCm'], y=df['PetalWidthCm'], data=df, ci=None, palette='hls')
plt.show()


# * Petal Length Vs Sepal Width

# In[20]:


plt.figure(figsize=(5,4))
sns.lineplot(x=df['PetalLengthCm'], y=df['SepalWidthCm'], data=df, ci=None, palette='hls')
plt.show()


# **Numerical Vs Target**

# In[21]:


for i in numerical_cols:
        plt.figure(figsize=(5,4))
        sns.lineplot(x=df['Species'], y=df[i], data=df, ci=None, palette='hls')
        plt.show()


# #### Analyzing Correlation Matrix

# In[22]:


lab=LabelEncoder()
df['Species_encoded']=lab.fit_transform(df['Species'])


# In[23]:


plt.figure(figsize=(6,4))
dataplot = sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f")  
plt.title('Correlation Plot')
plt.show()


# ### Model Building

# In[27]:


X = df.drop(['Species','Species_encoded'],axis=1)
y = df['Species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


# In[28]:


models = LogisticRegression()


# In[29]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
    
print(confusion_matrix(y_test,y_pred))
print()
print(classification_report(y_test,y_pred))


# **Model Summary:**

# * All models (Logistic Regression, Random Forest, Decision Tree, Gradient Boosting) achieved a perfect accuracy of 1, indicating excellent performance on the Iris dataset.

# In[31]:


import pickle


# In[32]:


model_filename='iris_model.pkl'
with open(model_filename,'wb') as model_file:
    pickle.dump(model,model_file)


# In[ ]:




