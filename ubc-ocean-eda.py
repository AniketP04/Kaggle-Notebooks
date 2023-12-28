#!/usr/bin/env python
# coding: utf-8

# # UBC-OCEAN

# ## Importing Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage import io
import numpy as np
import os, glob


# In[2]:


DATASET_FOLDER = "/kaggle/input/UBC-OCEAN/"
DATASET_IMAGES = "/kaggle/input/cancer-subtype-eda-load-wsi-prune-bg/train_thumbnails/"


# ## Load the train dataset...

# In[3]:


train_df = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))

print(f"Dataset/train size: {len(train_df)}")
display(train_df.head())


# In[4]:


print(train_df.info())


# ## Class Distribution

# In[5]:


plt.figure(figsize=(8, 6))
sns.countplot(x="label", data=train_df)
plt.title("Class Distribution")
plt.show()


# ## Image Dimensions Distribution

# In[6]:


plt.figure(figsize=(12, 6))
sns.scatterplot(x="image_width", y="image_height", hue="label", data=train_df)
plt.title("Image Dimensions Distribution")
plt.show()


# ## Class wise Thumbnail Image

# In[7]:


for lb, dfg in train_df.groupby("label"):
    fig, axes = plt.subplots(ncols=4, figsize=(16, 4))
    for i, name in enumerate(dfg["image_id"].sample(4)):
       
        img_path = os.path.join(DATASET_FOLDER, "train_thumbnails", f"{name}_thumbnail.png")
        
        if not os.path.isfile(img_path):
            img_path = os.path.join(DATASET_FOLDER, "train_images", f"{name}.png")
            print(f"Missing thumbnail for {img_path} but img exists {os.path.isfile(img_path)}")
            continue
            
        axes[i].imshow(plt.imread(img_path))
        axes[i].set_title(f"label: *{lb}* for img: {name}")
        axes[i].set_axis_off()
    fig.show()


# ## Tissue Microarray vs. Whole Slide Images

# In[8]:


plt.figure(figsize=(8, 6))
sns.countplot(x="is_tma", data=train_df)
plt.title("Tissue Microarray vs. Whole Slide Images")
plt.show()


# ## Image Dimensions Boxplot by Class

# In[9]:


plt.figure(figsize=(12, 6))
sns.boxplot(x="label", y="image_width", data=train_df)
plt.title("Image Width Distribution by Class")
plt.show()


# ## Image Dimensions Boxplot by Class

# In[10]:


plt.figure(figsize=(12, 6))
sns.boxplot(x="label", y="image_height", data=train_df)
plt.title("Image Height Distribution by Class")
plt.show()


# ## Pairplot

# In[11]:


sns.pairplot(train_df, hue="label")
plt.suptitle("Pairplot by Class", y=1.02)
plt.show()


# ## Distribution of Image Width

# In[12]:


plt.figure(figsize=(10, 6))
sns.histplot(train_df["image_width"], bins=30, kde=True)
plt.title("Distribution of Image Width")
plt.show()


# ## Distribution of Image Height

# In[13]:


plt.figure(figsize=(10, 6))
sns.histplot(train_df["image_height"], bins=30, kde=True)
plt.title("Distribution of Image Height")
plt.show()


# ## Distribution of Image Dimensions by Tissue Microarray

# In[14]:


plt.figure(figsize=(12, 6))
sns.scatterplot(x="image_width", y="image_height", hue="is_tma", data=train_df)
plt.title("Distribution of Image Dimensions by Tissue Microarray")
plt.show()


# ## Violin Plot of Image Dimensions by Class

# In[15]:


plt.figure(figsize=(14, 8))
sns.violinplot(x="label", y="image_width", data=train_df, inner="quartile")
plt.title("Violin Plot of Image Width by Class")
plt.show()


# In[16]:


plt.figure(figsize=(14, 8))
sns.violinplot(x="label", y="image_height", data=train_df, inner="quartile")
plt.title("Violin Plot of Image Height by Class")
plt.show()


# ## Pairplot of Image Dimensions by Class

# In[17]:


plt.figure(figsize=(12, 10))
sns.pairplot(train_df, hue="label", vars=["image_width", "image_height"])
plt.suptitle("Pairplot of Image Dimensions by Class", y=1.02)
plt.show()


# ## Distribution of Image Dimensions by Class and Tissue Microarray

# In[18]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x="image_width", y="image_height", hue="label", style="is_tma", data=train_df)
plt.title("Distribution of Image Dimensions by Class and Tissue Microarray")
plt.show()


# ## Samples: train_images

# In[19]:


for label in ['HGSC', 'CC', 'EC', 'LGSC', 'MC']:
    df_tmp = train_df[train_df['label']==label]
    image_id_list = list(df_tmp[df_tmp['is_tma']]['image_id'])
    plt.figure(figsize=(20.0, 6.0))
    
    for i in range(len(image_id_list)):
        image_id = image_id_list[i]
        plt.subplot(1, 5, i+1)
        if i == 0:
            plt.title(f'image_id:{image_id} (TMA)', fontsize=14)
            plt.ylabel(label, fontsize=14)
        else:
            plt.title(f'image_id:{image_id} (TMA)', fontsize=14)
        io.imshow(f'/kaggle/input/UBC-OCEAN/train_images/{image_id}.png')
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)


# ## Samples: train_thumbnails

# In[20]:


for label in ['HGSC', 'CC', 'EC', 'LGSC', 'MC']:
    df_tmp = train_df[train_df['label']==label]
    image_id_list = list(df_tmp[~df_tmp['is_tma']]['image_id'].sample(5))
    plt.figure(figsize=(20.0, 6.0))
    
    for i in range(len(image_id_list)):
        image_id = image_id_list[i]
        plt.subplot(1, 5, i+1)
        if i == 0:
            plt.title(f'image_id:{image_id} (WSI)', fontsize=14)
            plt.ylabel(label, fontsize=14)
        else:
            plt.title(f'image_id:{image_id} (WSI)', fontsize=14)
            
        io.imshow(f'/kaggle/input/UBC-OCEAN/train_thumbnails/{image_id}_thumbnail.png')
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.show()


# ## **Continue...**
