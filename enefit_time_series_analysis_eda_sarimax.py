# %% [markdown]
# # Enefit - Predict Energy Behavior of Prosumers

# %% [markdown]
# ### Importing Libraries

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:08:25.307759Z","iopub.execute_input":"2023-11-19T10:08:25.308171Z","iopub.status.idle":"2023-11-19T10:08:27.049597Z","shell.execute_reply.started":"2023-11-19T10:08:25.308112Z","shell.execute_reply":"2023-11-19T10:08:27.048389Z"}}
import os, glob
import json
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose

import warnings
warnings.filterwarnings("ignore")

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:08:27.051644Z","iopub.execute_input":"2023-11-19T10:08:27.052107Z","iopub.status.idle":"2023-11-19T10:08:27.057616Z","shell.execute_reply.started":"2023-11-19T10:08:27.052065Z","shell.execute_reply":"2023-11-19T10:08:27.056404Z"}}
PATH_DATASET = "/kaggle/input/predict-energy-behavior-of-prosumers"

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:08:27.058967Z","iopub.execute_input":"2023-11-19T10:08:27.0594Z","iopub.status.idle":"2023-11-19T10:08:39.735879Z","shell.execute_reply.started":"2023-11-19T10:08:27.059362Z","shell.execute_reply":"2023-11-19T10:08:39.73466Z"}}
forecast_weather_df  = pd.read_csv(os. path.join(PATH_DATASET, f"forecast_weather.csv"))
forecast_weather_df ['origin_datetime'] = pd.to_datetime(forecast_weather_df ['origin_datetime'])
forecast_weather_df ['forecast_datetime'] = pd.to_datetime(forecast_weather_df ['forecast_datetime'])
print(f"forecast size: {len(forecast_weather_df )}")

display(forecast_weather_df .head())

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:08:39.738057Z","iopub.execute_input":"2023-11-19T10:08:39.738421Z","iopub.status.idle":"2023-11-19T10:08:42.909106Z","shell.execute_reply.started":"2023-11-19T10:08:39.73839Z","shell.execute_reply":"2023-11-19T10:08:42.907913Z"}}
historical_weather_df = pd.read_csv(os. path.join(PATH_DATASET, f"historical_weather.csv"))
historical_weather_df ['datetime'] = pd.to_datetime(historical_weather_df ['datetime'])
print(f"histirical size: {len(historical_weather_df )}")

display(historical_weather_df .head())

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:08:42.9104Z","iopub.execute_input":"2023-11-19T10:08:42.910749Z","iopub.status.idle":"2023-11-19T10:08:44.391488Z","shell.execute_reply.started":"2023-11-19T10:08:42.910722Z","shell.execute_reply":"2023-11-19T10:08:44.390232Z"}}
train_df = pd.read_csv(os. path.join(PATH_DATASET,f'train.csv'))
gas_prices_df = pd.read_csv(os. path.join(PATH_DATASET,f'gas_prices.csv'))
client_df = pd.read_csv(os. path.join(PATH_DATASET,f'client.csv'))
electricity_prices_df = pd.read_csv(os. path.join(PATH_DATASET,f'electricity_prices.csv'))

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:08:44.393041Z","iopub.execute_input":"2023-11-19T10:08:44.39391Z","iopub.status.idle":"2023-11-19T10:08:44.441423Z","shell.execute_reply.started":"2023-11-19T10:08:44.393868Z","shell.execute_reply":"2023-11-19T10:08:44.440206Z"}}
print("Train Data:")
print(train_df.info())

print("\nGas Prices Data:")
print(gas_prices_df.info())

print("\nClient Data:")
print(client_df.info())

print("\nElectricity Prices Data:")
print(electricity_prices_df.info())

print("\nForecast Weather Data:")
print(forecast_weather_df .info())

print("\nHistorical Weather Data:")
print(historical_weather_df .info())

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:08:44.442709Z","iopub.execute_input":"2023-11-19T10:08:44.443008Z","iopub.status.idle":"2023-11-19T10:08:44.894953Z","shell.execute_reply.started":"2023-11-19T10:08:44.442982Z","shell.execute_reply":"2023-11-19T10:08:44.893833Z"}}
print("\nTrain Data Statistics:")
print(train_df.describe())

# %% [markdown]
# ### Prosumer Distribution by County

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:08:47.16737Z","iopub.execute_input":"2023-11-19T10:08:47.167766Z","iopub.status.idle":"2023-11-19T10:08:48.605272Z","shell.execute_reply.started":"2023-11-19T10:08:47.167737Z","shell.execute_reply":"2023-11-19T10:08:48.603855Z"}}
sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.histplot(data=train_df, x='county', hue='is_business', multiple='stack', bins=30, palette='viridis', alpha=0.7)

plt.xlabel('County')
plt.ylabel('Count')
plt.title('Prosumer Distribution by County')

plt.legend(title='Is Business', loc='upper right')

plt.show()

# %% [markdown]
# ### Energy Consumption vs. Production

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:08:48.607245Z","iopub.execute_input":"2023-11-19T10:08:48.608042Z","iopub.status.idle":"2023-11-19T10:08:54.53832Z","shell.execute_reply.started":"2023-11-19T10:08:48.608007Z","shell.execute_reply":"2023-11-19T10:08:54.537018Z"}}
energy_consumption_production = px.scatter(train_df, x='datetime', y='target', color='is_consumption',
                                          title='Energy Consumption vs. Production Over Time',
                                          labels={'datetime': 'Datetime', 'target': 'Energy Amount'},
                                          opacity=0.7)
energy_consumption_production.update_traces(marker=dict(size=5))

# %% [markdown]
# ### Gas Prices Over Time

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:09:22.302477Z","iopub.execute_input":"2023-11-19T10:09:22.302947Z","iopub.status.idle":"2023-11-19T10:09:22.310852Z","shell.execute_reply.started":"2023-11-19T10:09:22.302914Z","shell.execute_reply":"2023-11-19T10:09:22.309679Z"}}
with open(os. path.join(PATH_DATASET, f"county_id_to_name_map.json")) as fo:
    county_id_to_name = json.load(fo)
print(county_id_to_name)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:10:22.376251Z","iopub.execute_input":"2023-11-19T10:10:22.376662Z","iopub.status.idle":"2023-11-19T10:10:22.383804Z","shell.execute_reply.started":"2023-11-19T10:10:22.376632Z","shell.execute_reply":"2023-11-19T10:10:22.382405Z"}}
counties_locations = {
    "HARJUMAA": (59.351142, 24.725384),
    "HIIUMAA": (58.918082, 22.586403),
    "IDA-VIRUMAA": (59.228971, 27.406654),
    "JÄRVAMAA": (58.897934, 25.623048),
    "JÕGEVAMAA": (58.727941, 26.413961),
    "LÄÄNE-VIRUMAA": (59.267897, 26.363968),
    "LÄÄNEMAA": (58.975935, 23.772451),
    "PÄRNUMAA": (58.448793, 24.526469),
    "PÕLVAMAA": (58.089925, 27.101149),
    "RAPLAMAA": (58.924451, 24.619842),
    "SAAREMAA": (58.414075, 22.525137),
    "TARTUMAA": (58.394168, 26.747568),
    "VALGAMAA": (57.933466, 26.191360),
    "VILJANDIMAA": (58.316916, 25.595130),
    "VÕRUMAA": (57.765485, 27.025669)
}

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:10:24.51699Z","iopub.execute_input":"2023-11-19T10:10:24.517431Z","iopub.status.idle":"2023-11-19T10:10:24.630977Z","shell.execute_reply.started":"2023-11-19T10:10:24.517396Z","shell.execute_reply":"2023-11-19T10:10:24.630168Z"}}
cols_loc = ["latitude", "longitude"]
weather_locations = historical_weather_df.groupby(cols_loc).size().reset_index()[cols_loc]
display(weather_locations)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:10:26.857202Z","iopub.execute_input":"2023-11-19T10:10:26.857621Z","iopub.status.idle":"2023-11-19T10:10:26.98145Z","shell.execute_reply.started":"2023-11-19T10:10:26.857586Z","shell.execute_reply":"2023-11-19T10:10:26.980342Z"}}

map_ = folium.Map(location=[58.595272, 25.013607], zoom_start=7)

for county, coords in counties_locations.items():
    folium.CircleMarker(
        location=[coords[0], coords[1]], radius=5, color='cornflowerblue', fill=True
    ).add_child(folium.Popup(county)).add_to(map_)

for _, loc in weather_locations.iterrows():
    folium.CircleMarker(
        location=[loc["latitude"], loc["longitude"]], radius=1, color='orange'
    ).add_to(map_)
    
map_

# %% [markdown]
# ### Gas Prices Over Time

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:10:28.393398Z","iopub.execute_input":"2023-11-19T10:10:28.394076Z","iopub.status.idle":"2023-11-19T10:10:28.61853Z","shell.execute_reply.started":"2023-11-19T10:10:28.394033Z","shell.execute_reply":"2023-11-19T10:10:28.617309Z"}}
gas_prices_over_time = px.line(gas_prices_df, x='forecast_date', y=['lowest_price_per_mwh', 'highest_price_per_mwh'],
                               title='Gas Prices Over Time',
                               labels={'forecast_date': 'Forecast Date', 'value': 'Price (Euro/MWh)'})
gas_prices_over_time.update_layout(legend_title_text='Price Type')


# %% [markdown]
# ### Installed Capacity Distribution by Product Type

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:10:30.259839Z","iopub.execute_input":"2023-11-19T10:10:30.260264Z","iopub.status.idle":"2023-11-19T10:10:30.338199Z","shell.execute_reply.started":"2023-11-19T10:10:30.260231Z","shell.execute_reply":"2023-11-19T10:10:30.337169Z"}}
installed_capacity_distribution = px.box(client_df, x='product_type', y='installed_capacity',
                                         title='Installed Capacity Distribution by Product Type',
                                         labels={'product_type': 'Product Type', 'installed_capacity': 'Installed Capacity (KW)'})
installed_capacity_distribution.update_traces(boxpoints='all', jitter=0.3)

# %% [markdown]
# ### Electricity Prices Distribution

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:10:31.015802Z","iopub.execute_input":"2023-11-19T10:10:31.016583Z","iopub.status.idle":"2023-11-19T10:10:31.478266Z","shell.execute_reply.started":"2023-11-19T10:10:31.016542Z","shell.execute_reply":"2023-11-19T10:10:31.477107Z"}}
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(electricity_prices_df['euros_per_mwh'], bins=30, kde=False, color='skyblue', edgecolor='black')

plt.title('Electricity Prices Distribution', fontsize=16)
plt.xlabel('Electricity Price (Euro/MWh)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# ### Wind Speed Distribution

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:10:36.305611Z","iopub.execute_input":"2023-11-19T10:10:36.30599Z","iopub.status.idle":"2023-11-19T10:10:39.269509Z","shell.execute_reply.started":"2023-11-19T10:10:36.30596Z","shell.execute_reply":"2023-11-19T10:10:39.268428Z"}}
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(forecast_weather_df['10_metre_u_wind_component'], bins=30, kde=False, color='skyblue')

plt.title('Wind Speed Distribution', fontsize=16)
plt.xlabel('Wind Speed (m/s)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.show()

# %% [markdown]
# ### Historical Rainfall Over Time

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:10:40.777932Z","iopub.execute_input":"2023-11-19T10:10:40.778358Z","iopub.status.idle":"2023-11-19T10:11:03.241848Z","shell.execute_reply.started":"2023-11-19T10:10:40.778326Z","shell.execute_reply":"2023-11-19T10:11:03.240289Z"}}
historical_rainfall_over_time = go.Figure()
historical_rainfall_over_time.add_trace(go.Scatter(x=historical_weather_df['datetime'],
                                                   y=historical_weather_df['rain'],
                                                   mode='lines',
                                                   name='Rainfall'))
historical_rainfall_over_time.update_layout(title='Historical Rainfall Over Time',
                                            xaxis_title='Datetime',
                                            yaxis_title='Rainfall (mm)')


# %% [markdown]
# ### Prosumer Type Pie Chart

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:11:03.248594Z","iopub.execute_input":"2023-11-19T10:11:03.248899Z","iopub.status.idle":"2023-11-19T10:11:03.495558Z","shell.execute_reply.started":"2023-11-19T10:11:03.24887Z","shell.execute_reply":"2023-11-19T10:11:03.494636Z"}}
prosumer_type_counts = train_df['is_business'].value_counts()

plt.figure(figsize=(8, 8))
sns.set_palette("pastel")
plt.pie(prosumer_type_counts, labels=prosumer_type_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
plt.title('Prosumer Type Distribution')
plt.show()

# %% [markdown]
# ### Correlation Heatmap

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:11:48.38036Z","iopub.execute_input":"2023-11-19T10:11:48.380818Z","iopub.status.idle":"2023-11-19T10:11:48.980369Z","shell.execute_reply.started":"2023-11-19T10:11:48.380782Z","shell.execute_reply":"2023-11-19T10:11:48.978803Z"}}
cols = [c for c in train_df.columns if c not in ["datetime", "date"]]

correlation_matrix = train_df[cols].corr()

fig = px.imshow(correlation_matrix,
                labels=dict(x="Features", y="Features", color="Correlation"),
                x=cols,
                y=cols,
                color_continuous_scale="Viridis",
                title="Correlation Heatmap")

fig.show()

# %% [markdown]
# ### Gas Prices vs. Electricity Prices Over Time

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:11:50.615175Z","iopub.execute_input":"2023-11-19T10:11:50.61558Z","iopub.status.idle":"2023-11-19T10:11:50.673941Z","shell.execute_reply.started":"2023-11-19T10:11:50.615545Z","shell.execute_reply":"2023-11-19T10:11:50.672801Z"}}
prices_over_time = go.Figure()
prices_over_time.add_trace(go.Scatter(x=gas_prices_df['forecast_date'], y=gas_prices_df['lowest_price_per_mwh'],
                                      mode='lines', name='Gas Prices'))
prices_over_time.add_trace(go.Scatter(x=electricity_prices_df['forecast_date'],
                                      y=electricity_prices_df['euros_per_mwh'],
                                      mode='lines', name='Electricity Prices'))
prices_over_time.update_layout(title='Gas Prices vs. Electricity Prices Over Time',
                               xaxis_title='Forecast Date',
                               yaxis_title='Price (Euro/MWh)')

# %% [markdown]
# ## Time Series Analysis

# %% [markdown]
# ### Energy Consumption Over Time

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:12:02.916196Z","iopub.execute_input":"2023-11-19T10:12:02.916669Z","iopub.status.idle":"2023-11-19T10:12:07.608304Z","shell.execute_reply.started":"2023-11-19T10:12:02.916629Z","shell.execute_reply":"2023-11-19T10:12:07.607184Z"}}
subset_train_df = train_df[800:10000]

sns.set_theme()

plt.figure(figsize=(12, 6))
sns.lineplot(data=subset_train_df, x='datetime', y='target', hue='is_consumption', style='is_consumption', markers=True)

plt.xlabel('Datetime')
plt.ylabel('Energy Amount')
plt.title('Energy Consumption Over Time')

plt.legend(title='Is Consumption', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %% [markdown]
# ### Electricity Prices Over Time

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:12:07.610519Z","iopub.execute_input":"2023-11-19T10:12:07.610869Z","iopub.status.idle":"2023-11-19T10:12:07.707094Z","shell.execute_reply.started":"2023-11-19T10:12:07.610838Z","shell.execute_reply":"2023-11-19T10:12:07.705998Z"}}
electricity_prices_time_series = px.line(electricity_prices_df, x='forecast_date', y='euros_per_mwh',
                                         title='Electricity Prices Over Time',
                                         labels={'forecast_date': 'Forecast Date', 'euros_per_mwh': 'Electricity Price (Euro/MWh)'})
electricity_prices_time_series.show()

# %% [markdown]
# ### Historical Rainfall Over Time

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:12:10.999396Z","iopub.execute_input":"2023-11-19T10:12:10.999822Z","iopub.status.idle":"2023-11-19T10:12:33.569975Z","shell.execute_reply.started":"2023-11-19T10:12:10.999786Z","shell.execute_reply":"2023-11-19T10:12:33.568203Z"}}
historical_rainfall_time_series = px.line(historical_weather_df, x='datetime', y='rain',
                                          title='Historical Rainfall Over Time',
                                          labels={'datetime': 'Datetime', 'rain': 'Rainfall (mm)'})
historical_rainfall_time_series.show()

# %% [markdown]
# ### Temperature vs. Dewpoint Over Time

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:12:33.573079Z","iopub.execute_input":"2023-11-19T10:12:33.574179Z","iopub.status.idle":"2023-11-19T10:12:33.708811Z","shell.execute_reply.started":"2023-11-19T10:12:33.57409Z","shell.execute_reply":"2023-11-19T10:12:33.707362Z"}}
temp_dewpoint_comparison = px.scatter(forecast_weather_df[:1000], x='forecast_datetime', y='temperature',
                                      color='dewpoint', title='Temperature vs. Dewpoint Over Time',
                                      labels={'forecast_datetime': 'Forecast Datetime', 'temperature': 'Temperature (°C)', 'dewpoint': 'Dewpoint (°C)'})
temp_dewpoint_comparison.show()

# %% [markdown]
# ### Wind Speed vs. Solar Radiation Over Time

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:13:45.15736Z","iopub.execute_input":"2023-11-19T10:13:45.157746Z","iopub.status.idle":"2023-11-19T10:13:45.235675Z","shell.execute_reply.started":"2023-11-19T10:13:45.157718Z","shell.execute_reply":"2023-11-19T10:13:45.234382Z"}}
wind_speed_solar_radiation = px.scatter(forecast_weather_df[:1000], x='forecast_datetime', y='10_metre_u_wind_component',
                                        color='surface_solar_radiation_downwards',
                                        title='Wind Speed vs. Solar Radiation Over Time',
                                        labels={'forecast_datetime': 'Forecast Datetime', '10_metre_u_wind_component': 'Wind Speed (m/s)', 'surface_solar_radiation_downwards': 'Solar Radiation (W/m^2)'})
wind_speed_solar_radiation.show()

# %% [markdown]
# # TSA: SARIMAX

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:13:56.645747Z","iopub.execute_input":"2023-11-19T10:13:56.64695Z","iopub.status.idle":"2023-11-19T10:14:08.424202Z","shell.execute_reply.started":"2023-11-19T10:13:56.646911Z","shell.execute_reply":"2023-11-19T10:14:08.42301Z"}}
!pip install pmdarima
from pmdarima.arima import auto_arima

# %% [markdown]
# ### Importing Libraries

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:14:13.203813Z","iopub.execute_input":"2023-11-19T10:14:13.205887Z","iopub.status.idle":"2023-11-19T10:14:14.905313Z","shell.execute_reply.started":"2023-11-19T10:14:13.205841Z","shell.execute_reply":"2023-11-19T10:14:14.90356Z"}}
train = pd.read_csv('/kaggle/input/predict-energy-behavior-of-prosumers/train.csv')
train= train.dropna()
train= train.head(30000)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:14:16.396004Z","iopub.execute_input":"2023-11-19T10:14:16.396856Z","iopub.status.idle":"2023-11-19T10:14:16.417063Z","shell.execute_reply.started":"2023-11-19T10:14:16.396812Z","shell.execute_reply":"2023-11-19T10:14:16.416212Z"}}
train['datetime'] = pd.to_datetime(train['datetime'])
train.set_index('datetime', inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:14:16.970237Z","iopub.execute_input":"2023-11-19T10:14:16.971017Z","iopub.status.idle":"2023-11-19T10:14:16.985615Z","shell.execute_reply.started":"2023-11-19T10:14:16.970981Z","shell.execute_reply":"2023-11-19T10:14:16.984203Z"}}
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:14:17.522794Z","iopub.execute_input":"2023-11-19T10:14:17.523502Z","iopub.status.idle":"2023-11-19T10:14:17.528776Z","shell.execute_reply.started":"2023-11-19T10:14:17.523468Z","shell.execute_reply":"2023-11-19T10:14:17.527526Z"}}
y = train['target']

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:14:18.837837Z","iopub.execute_input":"2023-11-19T10:14:18.838246Z","iopub.status.idle":"2023-11-19T10:14:18.844314Z","shell.execute_reply.started":"2023-11-19T10:14:18.838212Z","shell.execute_reply":"2023-11-19T10:14:18.843097Z"}}
train_size = int(len(y) * 0.8)
train_set, test_set = y[:train_size], y[train_size:]


# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:14:20.34849Z","iopub.execute_input":"2023-11-19T10:14:20.349733Z","iopub.status.idle":"2023-11-19T10:14:20.927354Z","shell.execute_reply.started":"2023-11-19T10:14:20.349681Z","shell.execute_reply":"2023-11-19T10:14:20.926201Z"}}
plt.figure(figsize=(12, 8))
plt.plot(train_set[:500], label='Original Series')
plt.title('Original Series')


# %% [markdown]
# ### Seasonal Decomposition

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:14:23.049878Z","iopub.execute_input":"2023-11-19T10:14:23.05124Z","iopub.status.idle":"2023-11-19T10:14:24.888788Z","shell.execute_reply.started":"2023-11-19T10:14:23.051197Z","shell.execute_reply":"2023-11-19T10:14:24.887705Z"}}
result = seasonal_decompose(train_set, model='additive', period=12)  # Assuming seasonality is 24 hours
trend = result.trend.dropna()
seasonal = result.seasonal.dropna()
residual = result.resid.dropna()

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(train_set, label='Original Series')
plt.title('Original Series')

plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend')
plt.title('Trend Component')

plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonal')
plt.title('Seasonal Component')

plt.subplot(4, 1, 4)
plt.plot(residual, label='Residuals')
plt.title('Residual Component')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### ADF Test

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:14:28.495963Z","iopub.execute_input":"2023-11-19T10:14:28.49683Z","iopub.status.idle":"2023-11-19T10:14:31.060706Z","shell.execute_reply.started":"2023-11-19T10:14:28.496789Z","shell.execute_reply":"2023-11-19T10:14:31.059176Z"}}
result_adf = adfuller(train_set)
print(f'ADF Statistic: {result_adf[0]}')
print(f'p-value: {result_adf[1]}')
print('Critical Values:')
for key, value in result_adf[4].items():
    print(f'   {key}: {value}')

# %% [markdown]
# ### Plot: ACF and PACF

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:14:34.635084Z","iopub.execute_input":"2023-11-19T10:14:34.635492Z","iopub.status.idle":"2023-11-19T10:14:35.592193Z","shell.execute_reply.started":"2023-11-19T10:14:34.63546Z","shell.execute_reply":"2023-11-19T10:14:35.590887Z"}}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(train_set, lags=10, ax=ax1)
plot_pacf(train_set, lags=10, ax=ax2)
plt.show()


# %% [markdown]
# ### Auto- ARIMA

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:14:41.319844Z","iopub.execute_input":"2023-11-19T10:14:41.321225Z","iopub.status.idle":"2023-11-19T10:20:49.464156Z","shell.execute_reply.started":"2023-11-19T10:14:41.321173Z","shell.execute_reply":"2023-11-19T10:20:49.462921Z"}}
auto_arima_model = auto_arima(train_set, seasonal=True, m=10, trace=True)
print(auto_arima_model)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:20:59.918976Z","iopub.execute_input":"2023-11-19T10:20:59.923785Z","iopub.status.idle":"2023-11-19T10:20:59.942955Z","shell.execute_reply.started":"2023-11-19T10:20:59.923695Z","shell.execute_reply":"2023-11-19T10:20:59.94148Z"}}
order = auto_arima_model.order
print(order)
seasonal_order = auto_arima_model.seasonal_order
print(seasonal_order)

# %% [markdown]
# ### SARIMAX Model

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:21:02.558635Z","iopub.execute_input":"2023-11-19T10:21:02.559104Z","iopub.status.idle":"2023-11-19T10:21:08.140027Z","shell.execute_reply.started":"2023-11-19T10:21:02.559072Z","shell.execute_reply":"2023-11-19T10:21:08.138794Z"}}
model = SARIMAX(train_set, order=order, seasonal_order=seasonal_order)
result = model.fit(disp=False)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:21:08.141714Z","iopub.execute_input":"2023-11-19T10:21:08.142033Z","iopub.status.idle":"2023-11-19T10:21:08.355263Z","shell.execute_reply.started":"2023-11-19T10:21:08.142006Z","shell.execute_reply":"2023-11-19T10:21:08.353601Z"}}
predictions = result.get_forecast(steps=len(test_set))
predicted_mean = predictions.predicted_mean

# %% [markdown]
# ### Evaluation

# %% [code] {"execution":{"iopub.status.busy":"2023-11-19T10:21:10.6161Z","iopub.execute_input":"2023-11-19T10:21:10.616979Z","iopub.status.idle":"2023-11-19T10:21:10.625772Z","shell.execute_reply.started":"2023-11-19T10:21:10.616945Z","shell.execute_reply":"2023-11-19T10:21:10.624305Z"}}
mae = mean_absolute_error(test_set, predicted_mean)
print(f"Mean Absolute Error (MAE): {mae}")