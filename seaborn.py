#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("dark") # Theme for plots as Dark
# sns.set_theme(context='notebook', style='darkgrid', palette='deep')
print("Setup Complete")
sns.color_palette("flare")


# # **LINECHARTS & LINEPLOTS**
# To plot **Trends**
#  - sns.lineplot() - line charts 
#  - sns.violinplot() - voilin plots (combination of boxplot and kernel density estimate)

# In[2]:


museum_filepath = "../input/data-for-datavis/museum_visitors.csv"
museum_data = pd.read_csv(museum_filepath,index_col="Date",parse_dates = True)
museum_data.head(8)


# In[3]:



plt.figure(figsize=(16,6))
plt.title("Museum Data")
sns.lineplot(museum_data)
plt.show();


# In[4]:


plt.figure(figsize=(16,6))
sns.lineplot(museum_data['Chinese American Museum'])
sns.lineplot(museum_data['Firehouse Museum']);


# In[5]:


plt.figure(figsize=(16,6))
plt.subplot(2,1,1)
sns.lineplot(museum_data['Chinese American Museum'])
plt.subplot(2,1,2)
sns.lineplot(museum_data['Firehouse Museum']);


# In[6]:


plt.figure(figsize=(10,6))
sns.violinplot(data=museum_data,orient="v",palette="flare");


# # **BAR CHARTS**
# To plot **Relationships**
#  - sns.barplot() - Bar charts are useful for comparing quantities corresponding to different groups.
#  - sns.boxplot() - Box plots are useful for analysing std and outliers of the data
#  - sns.boxenplot() - Enhanced Box plot for larger datasets

# In[7]:


ign_filepath = "../input/data-for-datavis/ign_scores.csv"
ign_data = pd.read_csv(ign_filepath,index_col="Platform")
ign_data.head()


# In[8]:


plt.figure(figsize=(16,8))
sns.barplot(y=ign_data.index,x=ign_data["Racing"],palette="flare")
plt.xlabel("Rating")
plt.title("Average Score for Racing Games by Platform");


# In[9]:


plt.figure(figsize=(10,5))
sns.boxplot(ign_data,orient="h");


# In[10]:


plt.figure(figsize=(10,5))
sns.boxenplot(ign_data,orient="h");


# # **HEATMAPS**
# To plot **Relationships**
#  - sns.heatmap - Heatmaps can be used to find color-coded patterns in tables of numbers.

# In[11]:


plt.figure(figsize=(14, 10))
sns.heatmap(data=ign_data, annot=True);


# # **SCATTER PLOTS**
# To plot **Relationships**
#  - sns.scatterplot(): scatterplot <br>
#  - sns.regplot(): add regression line to scatterplot <br>
#  - sns.lmplot(): add multiple regression lines to scatterplot <br>
#  - sns.swarmplot(): categorical scatter plot<br>

# In[12]:


candy_filepath = "../input/data-for-datavis/candy.csv"
candy_data = pd.read_csv(candy_filepath,index_col="id")
candy_data.info()


# In[13]:


# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'
sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent']);


# In[14]:


# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'
sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent']);


# In[15]:


# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'
sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'],hue=candy_data['chocolate'])
plt.grid();


# In[16]:


# Color-coded scatter plot w/ regression lines
sns.lmplot(x="sugarpercent", y="winpercent", hue="chocolate", data=candy_data);


# In[17]:


# Scatter plot showing the relationship between 'chocolate' and 'winpercent'
sns.swarmplot(x = candy_data["chocolate"],y = candy_data["winpercent"]);


# # **HISTOGRAMS & DISTRIBUTIONS**
# To plot **Distributions**
#  - sns.histplot(): histogram <br>
#  - sns.kdeplot(): kernel density estimate (KDE) plot, a smoothed histogram <br>
#  - sns.jointplot(kind="kde"): two-dimensional (2D) KDE plot with the sns.jointplot <br>

# In[18]:


cancer_filepath = "../input/data-for-datavis/cancer.csv"
cancer_data = pd.read_csv(cancer_filepath,index_col="Id")
cancer_data.head()


# In[19]:


plt.figure(figsize=(16,8))
# Histograms for benign and maligant tumors
sns.histplot(cancer_data,x = "Area (mean)",hue="Diagnosis",bins=40);


# In[20]:


plt.figure(figsize=(16,8))
# KDE plots for benign and malignant tumors
sns.kdeplot(data=cancer_data,x="Radius (worst)",hue="Diagnosis",fill=True);


# In[21]:


plt.figure(figsize=(16,8));
sns.jointplot(x=cancer_data['Radius (mean)'], y=cancer_data['Texture (mean)'], kind="kde",fill=True,color ="salmon");

