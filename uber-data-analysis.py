#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv(r'UberDataset.csv')
df.head()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.describe()


# ## As of now there is only one numerical feature i.e MILES
# * maximum distance is 12204.7 mile
# * minimum is 0.5 miles
# * Avg distance is 21.115 miles

# ### These are the null values in each of the columns

# In[9]:


df.isnull().sum()


# In[11]:


df[df['END_DATE'].isnull()]


# ### As we can see this is a complete misinformation  so we could drop this row

# In[13]:


df.drop(1155,axis = 0, inplace =True)


# In[14]:


df.isnull().sum()


# In[16]:


df[df['PURPOSE'].isnull()]


# In[17]:


df["PURPOSE"] = df["PURPOSE"].fillna("Missing")


# In[18]:


df.isnull().sum()


# In[19]:


df.head()


# In[20]:


df['START_DATE'] = pd.to_datetime(df['START_DATE'])
df['END_DATE'] = pd.to_datetime(df['END_DATE'])


# In[21]:


df.head()


# ## Calculate the total duration of the Uber ride

# In[33]:


(df['END_DATE']-df['START_DATE'])


# In[34]:


df['Duration'] = (df['END_DATE']-df['START_DATE']).dt.total_seconds()/60


# In[42]:


df.head()


# ## Day of the week and month

# In[38]:


df['start_day_of_week'] = df['START_DATE'].dt.day_name()
df['end_day_of_week'] = df['END_DATE'].dt.day_name()


# In[44]:


df['Month'] = df['START_DATE'].dt.month_name()


# In[45]:


df.head()


# ## To calculate the time of a day

# In[47]:


def timeofdayfnc(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 20:
        return 'Evening'
    else:
        return 'Night'
df['time_of_day'] = df['START_DATE'].dt.hour.apply(timeofdayfnc)


# In[50]:


df.head(2)


# ## Now we will drop the original date time columns

# In[51]:


df.drop(columns = ['START_DATE','END_DATE'], axis = 1, inplace = True)


# In[52]:


df.head()


# # Data Visualisation

# In[54]:


sns.histplot(data = df, x = 'PURPOSE', kde = True)
plt.xticks(rotation = 90)


# ### Insights
# * Most of the customers are not willing to tell the actual purpose for their ride
# * In those who tell the purpose Meeting and Entertain is the most

# In[55]:


sns.histplot(data= df, x = 'CATEGORY')


# ### Insights
# * Most of the rides used for Business Category

# In[57]:


sns.histplot(data = df , x = 'start_day_of_week', kde = True)
plt.xticks(rotation = 50)


# ### Insights 
# * Most of the rides were used in Friday

# In[62]:


sns.histplot(data = df, x = 'time_of_day', kde=True)


# ### Insights
# * Most rides were used in the afternoon

# In[65]:


sns.histplot(data = df, x = 'Month',kde = True)
plt.xticks(rotation = 90)


# ### Pdf s of Miles and Duration

# In[77]:


sns.distplot(df['MILES'], hist = False)
sns.distplot(df['Duration'], hist = False)
plt.title('Miles Vs Duration')
plt.legend(['Miles','Duration'])


# In[78]:


sns.barplot(x= df["PURPOSE"], y= df['MILES'],hue = df["CATEGORY"])
plt.title("Purpose VS Miles Travelled By Category ")
plt.xticks(rotation=90)


# In[97]:


plt.figure(figsize=(20, 7))
sns.boxplot(data = df, x = df['MILES'], y = df['PURPOSE'])


# ### Insights
# * We can see that our MILES data consists many outliers.
# * Customers who have Errand/Supplies purpose tend to have relatively low travel distance.
# * On median, customers who have Meeting purpose travels further than most of the other purposes.
# * Customers who have Customer Visit purpose have relatively high variability than the other purposes

# In[90]:


plt.figure(figsize=(20, 10))
sns.boxplot(data = df, x = df['Duration'], y = df['PURPOSE'], hue = 'CATEGORY')


# ### Insights
# * We can see that our trip duration data also consists many outliers.
# * On median, customers who have Errand/Supplies purpose tend to have relatively low travel duration.
# * Customers with Meal/Entertain purpose have relatively low travel duration compared to other purposes in Business category.
# * On median, customers who have Meeting, Customer Visit, Temporary Site, and Between Offices purpose have similar travel duration.
# * Customers who have Customer Visit purpose have relatively high variability than the other purposes.

# ### Top 5 pickup spots

# In[171]:


fig, axs = plt.subplots(2,2, figsize = (16,6))
sns.barplot(data = df[(df['PURPOSE']=='Meeting')].groupby('START').size().reset_index(name = 'count').sort_values(by = 'count', ascending = False)[:5], x = 'START', y = 'count',ax = axs[0,0])
sns.barplot(data = df[(df['PURPOSE']=='Meal/Entertain')].groupby('START').size().reset_index(name = 'count').sort_values(by = 'count', ascending = False)[:5], x = 'START', y = 'count',ax = axs[0,1])
sns.barplot(data = df[(df['PURPOSE']=='Errand/Supplies')].groupby('START').size().reset_index(name = 'count').sort_values(by = 'count', ascending = False)[:5], x = 'START', y = 'count',ax = axs[1,0])
sns.barplot(data = df[(df['PURPOSE']=='Customer Visit')].groupby('START').size().reset_index(name = 'count').sort_values(by = 'count', ascending = False)[:5], x = 'START', y = 'count',ax = axs[1,1])


# ## Top 5 drop-off spots

# In[172]:


fig, axs = plt.subplots(2,2, figsize = (16,6))
sns.barplot(data = df[(df['PURPOSE']=='Meeting')].groupby('STOP').size().reset_index(name = 'count').sort_values(by = 'count', ascending = False)[:5], x = 'STOP', y = 'count',ax = axs[0,0])
sns.barplot(data = df[(df['PURPOSE']=='Meal/Entertain')].groupby('STOP').size().reset_index(name = 'count').sort_values(by = 'count', ascending = False)[:5], x = 'STOP', y = 'count',ax = axs[0,1])
sns.barplot(data = df[(df['PURPOSE']=='Errand/Supplies')].groupby('STOP').size().reset_index(name = 'count').sort_values(by = 'count', ascending = False)[:5], x = 'STOP', y = 'count',ax = axs[1,0])
sns.barplot(data = df[(df['PURPOSE']=='Customer Visit')].groupby('STOP').size().reset_index(name = 'count').sort_values(by = 'count', ascending = False)[:5], x = 'STOP', y = 'count',ax = axs[1,1])


# In[ ]:




