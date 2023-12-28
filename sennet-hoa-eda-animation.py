#!/usr/bin/env python
# coding: utf-8

# # SenNet + HOA

# ## Importing Libraries

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
from PIL import Image
import numpy as np

from matplotlib import animation, rc
rc('animation', html='jshtml')


# In[2]:


df = pd.read_csv('/kaggle/input/blood-vessel-segmentation/train_rles.csv')
df['subset']=df['id'].map(lambda x:'_'.join(x.split('_')[:-1]))
df.head()


# In[3]:


print("Number of unique slices:", df['id'].nunique())
print("Number of unique datasets:", df['id'].str.split('_').str[0].nunique())


# ## Distribution of Segmentation Mask Lengths

# In[4]:


df['mask_length'] = df['rle'].apply(lambda x: len(str(x).split()))

fig = px.histogram(df, x='mask_length', nbins=20,
                   labels={'mask_length': 'Number of Pixels in Mask', 'count': 'Frequency'},
                   title='Distribution of Segmentation Mask Lengths',
                   marginal="box",  # or violin, rug
                   color_discrete_sequence=['skyblue'])

fig.update_layout(bargap=0.1)
fig.show()


# In[5]:


df['scan'] = df['id'].apply(lambda x: x[:-5])
df['mask_is_empty'] = df['id'].apply(lambda x: x[:-5])

df['scan'] = df['id'].apply(lambda x: x[:-5])
df['mask_is_empty'] = df['rle']=='1 0'


# ## Number of Images per Scan

# In[6]:




fig = px.histogram(df, y='scan', title='Number of Images per Scan', labels={'scan': 'Scan'}, color_discrete_sequence=['skyblue'])
fig.update_layout(yaxis_title='Scan', xaxis_title='Number of Images')
fig.show()


# ## Mask is Empty Percentage

# In[7]:


x = 'scan'
y = 'mask_is_empty'

gb = df.groupby(x)[y].value_counts(normalize=True)
gb = gb.round(3)*100
gb = gb.rename('percent').reset_index()

fig = px.bar(gb, 
             x='percent',
             y='scan',
             color='mask_is_empty',
             orientation='h',
             title='Mask is Empty Percentage',
             labels={'scan': 'Scan', 'percent': 'Percentage'},
             category_orders={'scan': sorted(df['scan'].unique())},  
             color_discrete_sequence=px.colors.qualitative.Set1 
            )

for trace in fig.data:
    trace['text'] = [f"{round(val, 1)}%" for val in trace['x']]

fig.show()


# In[8]:


def load_image(file_path):
    with Image.open(file_path) as img:
        return np.array(img)


# In[9]:


def normalize_intensity(image):
    return image / 255.0


# In[10]:


def show(sample_df, idx):
    sample = sample_df[sample_df['slice_id'] == str.zfill(f'{idx}', 4)]

    image = load_image(sample['image'].values[0])
    label = load_image(sample['label'].values[0])

    image = normalize_intensity(image)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(image, cmap='gray')
    ax2.imshow(label, cmap='gray')
    ax1.axis('off')
    ax2.axis('off')
    plt.subplots_adjust(wspace=0.05)
    plt.show()


# In[11]:


def animate(sample_df, id_range):
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.axis('off')
    ax2.axis('off')
    images = []

    for i in tqdm(id_range):
        sample = sample_df[sample_df['slice_id'] == str.zfill(f'{i}', 4)]

        image = load_image(sample['image'].values[0])
        label = load_image(sample['label'].values[0])

        image = normalize_intensity(image)
    

        im1 = ax1.imshow(image, animated=True, cmap='gray')
        im2 = ax2.imshow(label, animated=True, cmap='gray')
        
        if i == id_range[0]:
            ax1.imshow(image, cmap='gray')
            ax2.imshow(label, cmap='gray')
        
        images.append([im1, im2])

    ani = animation.ArtistAnimation(fig, images, interval=50, blit=True, repeat_delay=1000)
    plt.close()
    return ani


# In[12]:


def prepare_df(im_dir='kidney_1_dense',lb_dir='kidney_1_dense'):
    df = pd.read_csv('/kaggle/input/blood-vessel-segmentation/train_rles.csv')
    base_dir = Path('/kaggle/input/blood-vessel-segmentation/train')
    subset_df = df[df.id.str.startswith(lb_dir)].reset_index(drop=True)
    subset_df['slice_id'] = subset_df['id'].map(lambda x:x.split('_')[-1]) 
    subset_df['image'] = subset_df['slice_id'].map(lambda x: base_dir / im_dir / 'images' / f'{x}.tif')
    subset_df['label'] = subset_df['slice_id'].map(lambda x: base_dir / lb_dir / 'labels' / f'{x}.tif')
    return subset_df


# ## kidney_1_dense

# In[13]:


dense_1_df = prepare_df(im_dir='kidney_1_dense',lb_dir='kidney_1_dense')
show(dense_1_df,1234)


# In[14]:


animate(dense_1_df,id_range=range(1200,1300))


# ## kidney_1_voi

# In[15]:


kidney_1_voi_df = prepare_df(im_dir='kidney_1_voi',lb_dir='kidney_1_voi')
show(kidney_1_voi_df,454)


# In[16]:


animate(kidney_1_voi_df,id_range=range(500,600))


# ## kidney_2

# In[17]:


kidney_2_df = prepare_df(im_dir='kidney_2',lb_dir='kidney_2')
show(kidney_2_df,1234)


# In[18]:


animate(kidney_2_df,id_range=range(800,900))


# ## kidney_3_dense

# In[19]:


kidney_3_dense_df = prepare_df(im_dir='kidney_3_sparse',lb_dir='kidney_3_dense')
show(kidney_3_dense_df,533)


# In[20]:


animate(kidney_3_dense_df,id_range=range(500,600))


# ## kidney_3_sparse

# In[21]:


kidney_3_sparse_df = prepare_df(im_dir='kidney_3_sparse',lb_dir='kidney_3_sparse')
show(kidney_3_sparse_df,343)


# In[22]:


animate(kidney_3_sparse_df,id_range=range(300,400))


# ## **Continue...**
