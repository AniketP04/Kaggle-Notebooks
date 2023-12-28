#!/usr/bin/env python
# coding: utf-8

# # **Importing Libraries**

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # **Reading data**

# In[6]:


wf=pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")


# In[7]:


wf.head()


# In[8]:


wf.shape


# # **Plotting**

# In[19]:


plt.figure(figsize = (4,4))
sns.countplot(x=wf['quality'])
plt.show()


# In[13]:


plt.figure(figsize=(4,4))
sns.barplot(x='quality', y='alcohol', data=wf, palette='inferno')
plt.show()


# In[14]:


plt.figure(figsize=(4,4))
sns.scatterplot(x='citric acid', y='pH', data=wf)
plt.show()


# In[16]:


plt.figure(figsize=(12,6))
sns.pairplot(wf)
plt.show()


# In[22]:


plt.figure(figsize=(12,6))
sns.heatmap(wf.corr(), annot=True)
plt.show()


# In[26]:


x=wf.drop(['quality'], axis=1)
y=wf['quality']
print("Done")


# # **Data Preprocessing**

# In[28]:


#handle unbalanced data
#to do this, we use oversampling
from imblearn.over_sampling import SMOTE
oversampling=SMOTE()
x_res,y_res=os.fit_resample(x, y)


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_res,y_res,test_size=0.2, random_state=0)


# In[30]:


from sklearn.preprocessing import StandardScaler

stdscale = StandardScaler().fit(x_train)
x_train_std = stdscale.transform(x_train)
x_test_std = stdscale.transform(x_test)


# In[31]:


from sklearn.metrics import accuracy_score


# # **Logistic Regression**

# In[32]:


#this is for classification
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_std, y_train)
predictions = lr.predict(x_test_std)
accuracy_score(y_test, predictions)


# # **Decision Tree Classifier**

# In[33]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train_std, y_train)
accuracy_score(y_test, dt.predict(x_test_std))


# # **Random Forest Classifier**

# In[34]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)
rf.fit(x_train_std, y_train)
accuracy_score(y_test, rf.predict(x_test_std))

