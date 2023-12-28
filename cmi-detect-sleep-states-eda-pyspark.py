#!/usr/bin/env python
# coding: utf-8

# # Child Mind Institute - Detect Sleep States

# ## Importing Libraries

# In[1]:


get_ipython().system('pip install pyspark')


# In[2]:


get_ipython().system('nvidia-smi')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions
from pyspark.sql import Row, SQLContext, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import col

import cudf
import multiprocessing
import os
import re
import time
import pynvml
import subprocess
import psutil

from warnings import filterwarnings;
filterwarnings('ignore');


# In[4]:


def get_time(func):
    def inner_get_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000 
        print("-" * 80)
        print(f"Execution time: {execution_time} ms")
        return result, execution_time  

    return inner_get_time


# In[5]:


def reduce_mem_usage(data):
    for col_name, col_data_type in data.dtypes:
        if col_data_type.startswith("int"):
            data = data.withColumn(col_name, col(col_name).cast("int"))
        elif col_data_type.startswith("float"):
            data = data.withColumn(col_name, col(col_name).cast("float"))
        elif col_data_type.startswith("decimal"):
            # Adjust the scale and precision as needed
            data = data.withColumn(col_name, col(col_name).cast("decimal(10, 2)"))
        elif col_data_type.startswith("timestamp"):
            data = data.withColumn(col_name, col(col_name).cast("timestamp"))
        elif col_data_type.startswith("string"):
            data = data.withColumn(col_name, col(col_name).cast("string"))
        elif col_data_type.startswith("boolean"):
            data = data.withColumn(col_name, col(col_name).cast("boolean"))
        elif col_data_type.startswith("array"):
            data = data.withColumn(col_name, col(col_name).cast("array<your_element_type>"))
        elif col_data_type.startswith("map"):
            data = data.withColumn(col_name, col(col_name).cast("map<your_key_type, your_value_type>"))
        

    return data


# In[6]:


def get_free_gpu_memory():
    try:
        pynvml.nvmlInit()

        gpu_count = pynvml.nvmlDeviceGetCount()

        gpu_memory_list = []

        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory_mb = gpu_memory.free
            return free_memory_mb

        pynvml.nvmlShutdown()

        return gpu_memory_list

    except Exception as e:
        return str(e)


# In[7]:


def convert_bytes(file_size):
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
   
    unit_index = 0
    while file_size >= 1024 and unit_index < len(units) - 1:
        file_size /= 1024.0
        unit_index += 1
  
    return f"{file_size:.2f} {units[unit_index]}"

def getSize(file):
    try:
        file_size = os.path.getsize(file)
        file_name = re.search(r'([^/\\]+)$', file).group(1)

        print(f"File {file_name} size is {convert_bytes(file_size)}")
    except FileNotFoundError:
        print("File not found.")
    except OSError:
        print("OS error occurred.")


# In[8]:


def get_cpu_core_count():
    try:
       
        cpu_core = multiprocessing.cpu_count()

        return cpu_core
    except Exception as e:
        return None

def get_gpu_core_count():
    try:
        pynvml.nvmlInit()

        gpu_count = pynvml.nvmlDeviceGetCount()

        pynvml.nvmlShutdown()

        return gpu_count

    except Exception as e:
        return None


# In[9]:


@get_time
def readParquet(file):
    data = spark_1.read         .option("header", "true")         .option("inferSchema", "true")         .parquet(file)         .cache()
    print(f"Number of Partitions -> {data.rdd.getNumPartitions()}")
    data.write.format("noop").mode("overwrite").save()
    return data


# In[10]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### PySpark Configurations

# In[11]:


gpu_memory = get_free_gpu_memory()
cpu_cores = get_cpu_core_count()
gpu_cores = get_gpu_core_count()

master = 'local[*]'
maxMemory = '8G'
app_name = 'Child Mind Institute Detect Sleep States'
maxBytes= str(gpu_memory) + "b"

conf = {
    'spark.dynamicAllocation.enabled': 'true',
    'spark.dynamicAllocation.maxExecutors': 8,
    
    'spark.shuffle.service.enabled': 'true',
    
    'spark.memory.fraction':'0.6',
    
    'spark.executor.memoryOverhead':maxMemory,
    'spark.executor.memory': maxMemory,
    'spark.executor.cores': cpu_cores,
    'spark.executor.resource.gpu.amount': gpu_cores * 3,
    'spark.executor.resource.gpu.memoryFraction':'0.6',
    
    'spark.rapids.memory.pinnedPool.size': maxBytes,
    'spark.rapids.sql.enabled': 'true',
    
    'spark.driver.memory': maxMemory,
    'spark.driver.maxResultsSize': maxMemory,
    
    'spark.sql.repl.eagerEval.enabled': 'true',
    'spark.sql.files.maxPartitionBytes': maxBytes,
    'spark.sql.files.maxPartitionNum': cpu_cores * 3,
    'spark.sql.files.minPartitionNum': cpu_cores * 2,
    'spark.sql.adaptive.enabled': 'false',
    'spark.sql.adaptive.localShuffleReader.enabled': 'false',
    'spark.sql.adaptive.coalescePartitions.enabled': 'false',
    'spark.sql.sources.partitionOverwriteMode': 'dynamic',
    'spark.sql.shuffle.partitions': '4',
}

builder = (
    SparkSession
    .builder
    .master(master)
    .appName(app_name)
)

for k, v in conf.items():
    builder.config(k, v)


# In[12]:


spark_1 = builder.getOrCreate()
spark_1.sparkContext.setLogLevel("ERROR")
spark_1


# In[13]:


train_series_dataset='../input/child-mind-institute-detect-sleep-states/train_series.parquet'
sample_submission_dataset='../input/child-mind-institute-detect-sleep-states/sample_submission.csv'
train_events_dataset='../input/child-mind-institute-detect-sleep-states/train_events.csv'
test_series_dataset='../input/child-mind-institute-detect-sleep-states/test_series.parquet'


# In[14]:


print(getSize(train_series_dataset))
print(getSize(sample_submission_dataset))
print(getSize(train_events_dataset))
print(getSize(test_series_dataset))


# ### Parquet File: train_series.parquet

# In[15]:


train_series, execution_time = readParquet(train_series_dataset)
train_series


# In[16]:


cleaned_train_series = reduce_mem_usage(train_series.na.drop())


# In[17]:


cleaned_train_series = cleaned_train_series         .withColumn("date", date_format("timestamp", "yyyy-MM-dd"))         .withColumn("time", date_format("timestamp", "hh:mm:ss"))         .withColumn("hour", date_format("timestamp", "hh"))         .withColumn("hour", col("hour").cast("int"))


# In[18]:


get_ipython().system('nvidia-smi')


# In[19]:


class AnalyseData:
    def __init__(self, data):
        self.data = data
    
    @get_time
    def countData(self):
        return self.data.count()

    @get_time
    def describeData(self):
        return self.data.describe()
    
    @get_time
    def printSchemaData(self):
        return self.data.printSchema()


# In[20]:


train_series_analyzer = AnalyseData(cleaned_train_series)


# In[21]:


count, count_time = train_series_analyzer.countData()
description, describe_time = train_series_analyzer.describeData()
printSchema, printSchema_time = train_series_analyzer.printSchemaData()


# In[22]:


print("Count:", count)
print("Execution time for countData+:", count_time, "ms")


# In[23]:


#print("Description:", description)
#print("Execution time for describeData:", describe_time, "ms")


# In[24]:


print("printSchema:", printSchema)
print("Execution time for printSchema:", printSchema_time, "ms")


# **Taken only 100 samples of 50% data for EDA**

# In[25]:


sampled_data = cleaned_train_series.sample(0.5).limit(100)


# ### Distribution of steps

# In[26]:


plt.figure(figsize=(12, 6))
sns.histplot(sampled_data.select('step').toPandas(), bins=50, kde=True)
plt.title('Distribution of Steps')
plt.xlabel('Step')
plt.ylabel('Frequency')
plt.show()


# ### Time series plot of anglez

# In[27]:


anglez_data = sampled_data.select("timestamp", "anglez").toPandas()
plt.figure(figsize=(12, 6))
plt.plot(anglez_data['timestamp'], anglez_data['anglez'])
plt.title('Time Series Plot of Anglez')
plt.xlabel('Timestamp')
plt.ylabel('Anglez')
plt.show()


# ### Time series plot of ENMO

# In[28]:


enmo_data = sampled_data.select("timestamp", "enmo").toPandas()
plt.figure(figsize=(12, 6))
plt.plot(enmo_data['timestamp'], enmo_data['enmo'])
plt.title('Time Series Plot of ENMO')
plt.xlabel('Timestamp')
plt.ylabel('ENMO')
plt.show()


# ### Boxplot of Anglez by Hour

# In[29]:


boxplot_anglez_hour_data = sampled_data.select("hour", "anglez").toPandas()
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='anglez', data=boxplot_anglez_hour_data)
plt.title('Boxplot of Anglez by Hour')
plt.xlabel('Hour')
plt.ylabel('Anglez')
plt.show()


# ### Boxplot of ENMO by Hour

# In[30]:


boxplot_enmo_hour_data = sampled_data.select("hour", "enmo").toPandas()
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='enmo', data=boxplot_enmo_hour_data)
plt.title('Boxplot of ENMO by Hour')
plt.xlabel('Hour')
plt.ylabel('ENMO')
plt.show()


# ### Scatter Plot between Anglez and ENMO

# In[31]:


scatter_data = sampled_data.select("anglez", "enmo").toPandas()
plt.figure(figsize=(12, 6))
sns.scatterplot(x='anglez', y='enmo', data=scatter_data)
plt.title('Scatter Plot between Anglez and ENMO')
plt.xlabel('Anglez')
plt.ylabel('ENMO')
plt.show()


# ### Violin Plot of Anglez

# In[32]:


violin_anglez_data = sampled_data.select("anglez").toPandas()
plt.figure(figsize=(12, 6))
sns.violinplot(x=violin_anglez_data['anglez'])
plt.title('Violin Plot of Anglez')
plt.xlabel('Anglez')
plt.show()


# ### Violin Plot of ENMO

# In[33]:


violin_enmo_data = sampled_data.select("enmo").toPandas()
plt.figure(figsize=(12, 6))
sns.violinplot(x=violin_enmo_data['enmo'])
plt.title('Violin Plot of ENMO')
plt.xlabel('ENMO')
plt.show()


# ### Pairplot of Anglez, ENMO, and Step

# In[34]:


pairplot_data = sampled_data.select("anglez", "enmo", "step").toPandas()
sns.pairplot(pairplot_data)
plt.suptitle('Pairplot of Anglez, ENMO, and Step', y=1.02)
plt.show()


# ### Correlation Matrix

# In[35]:


numerical_columns = ["step", "anglez", "enmo", "hour"]
correlation_matrix = sampled_data.select(numerical_columns).toPandas().corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# ## CSV File: train_events.csv

# In[36]:


train_events = pd.read_csv('/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv')
train_events


# In[37]:



train_events.shape


# In[38]:


plt.figure(figsize=(4,4))

ax = sns.countplot(data=train_events, x='event')
ax.bar_label(ax.containers[0])
plt.show()


# In[39]:


plt.figure(figsize=(6,4))

ax = plt.plot(list(train_events['night']))
plt.show()


# ### **Continue...**
