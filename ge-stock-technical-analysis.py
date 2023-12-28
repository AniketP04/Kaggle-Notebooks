#!/usr/bin/env python
# coding: utf-8

# # GE Stock - Technical Analysis

# Importing necessacy libraries

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px


# Reading dataset

# In[2]:


#setting date column as index
df=pd.read_csv(r"X:\Project\ETH\EthereumPriceNet-master\GE.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df.set_index('Date',inplace=True)


# Only 5% of data has been taken for the visualization.

# In[3]:


data = int(len(df) * 0.05)
data_viz = df[:data]
data_viz


# # Candlestick chart 

# In[4]:


#plotting candlestick plot 

fig = go.Figure(data=[go.Candlestick(x=data_viz.index,
                       open=data_viz.Open, high=data_viz.High,
                       low=data_viz.Low, close=data_viz.Close)])

fig.show()


# # 50, 100, 200 Moving Average 

# In[5]:


#calculating moving average
Moving_Average_Day = [50, 100, 200]
for Moving_Average in Moving_Average_Day:
  for company in df:
    column_name = f'Moving Average for {Moving_Average} days'
    data_viz[column_name] = data_viz["Close"].rolling(Moving_Average).mean()


# In[6]:


#plotting moving average
plt.figure(figsize=(20,8))
plt.plot(data_viz.index, data_viz["Close"])
plt.plot(data_viz.index, data_viz["Moving Average for 50 days"],color='red',label='MA for 50 days')
plt.plot(data_viz.index, data_viz["Moving Average for 100 days"],color='green',label='MA for 100 days')
plt.plot(data_viz.index, data_viz["Moving Average for 200 days"],color='orange',label='MA for 200 days')
plt.legend()


# # Bollinger Band

# In[7]:


#plotting bollinger band
rolling_mean = data_viz['Close'].rolling(window=20).mean()
rolling_std = data_viz['Close'].rolling(window=20).std()
upper_band = rolling_mean + (rolling_std * 2)
lower_band = rolling_mean - (rolling_std * 2)
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(data_viz.index, data_viz['Close'], label='Close')
ax.plot(rolling_mean.index, rolling_mean, label='Rolling Mean')
ax.fill_between(rolling_mean.index, upper_band, lower_band, alpha=0.4, color='gray', label='Bollinger Bands')
ax.legend()
plt.show()


# # Multiple Time Frame Chart

# In[9]:



fig = px.line(data_viz, x=data_viz.index, y='Close', title='Mulitple Time Frame Range Slider and Selectors')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=5, label="5s", step="second", stepmode="backward"),
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=1, label="1H", step="hour", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()


# # Updated Dataframe

# In[18]:


data_viz.head(100)

