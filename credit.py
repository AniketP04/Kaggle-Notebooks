#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt


import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator


# In[2]:


#initializing h2o
h2o.init()


# In[3]:


#uplaoding file
df = h2o.upload_file(r"X:\Project\Autoencoderr\New folder\creditcard.csv")


# In[4]:


df= df.drop(['Time'], axis=1)


# In[5]:


df.shape


# In[6]:


df


# In[7]:


#splitting data into train and test
train,test = df.split_frame(ratios=[.80])


# In[8]:


print(train.shape)
print(test.shape)


# In[9]:



train = train.drop(['Class'], axis=1)

y_test = test['Class']
y_test_df = y_test.as_data_frame()

test = test.drop(['Class'], axis=1)


# In[10]:


#h2o autoencoder
Autoencoder = h2o.estimators.deeplearning.H2OAutoEncoderEstimator(hidden=[20,15,10,15,20], 
                                                                  epochs=100, 
                                                                  activation='tanh', 
                                                                  autoencoder=True) 
Autoencoder.train(x = train.columns, training_frame = train) 


# In[22]:


print(Autoencoder)


# In[21]:


#predictions
prediction = Autoencoder.predict(test)


# In[13]:


#model loss
scoring_history = Autoencoder.score_history()
plt.plot(scoring_history['training_mse'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')


# In[14]:


#anomaly is a function which calculates the error for the dataset
test_rec_error = Autoencoder.anomaly(test)
test_rec_error_df = test_rec_error.as_data_frame()


# In[15]:


test_rec_error_df


# In[16]:


error_df = pd.DataFrame({'reconstruction_error': test_rec_error_df['Reconstruction.MSE'],
                        'true_class': y_test_df['Class']})


# In[17]:


error_df


# In[18]:


error_df.describe()


# In[19]:


from sklearn.metrics import ( auc,roc_curve,roc_auc_score)


fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)


plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[ ]:




