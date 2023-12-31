#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/nlp project')


# In[ ]:


pip install emoji


# In[ ]:


import emoji
import numpy as np
import pandas as pd


# In[ ]:


mapping = pd.read_csv("Mapping.csv")
output = pd.read_csv("OutputFormat.csv")
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")


# In[ ]:


mapping


# In[ ]:


mapping = mapping.drop("Unnamed: 0", axis=1)
mapping.head()


# In[ ]:


mapping = mapping.set_index('number')


# In[ ]:


mapping.head()


# In[ ]:


mapping_dict = mapping.to_dict()
mapping_dict = mapping_dict['emoticons']
mapping_dict


# In[ ]:


train.head()


# In[ ]:


train.columns


# In[ ]:


train = train.drop('Unnamed: 0', axis=1)


# In[ ]:


train.tail()


# In[ ]:


train.head()


# In[ ]:


X_train = train['TEXT'].values
y_train = train['Label'].values

print(X_train.shape, y_train.shape)


# In[ ]:


file = open("glove.6B.50d.txt", encoding = 'utf8')


# In[ ]:


def intialize_emb_matrix(file):
    embedding_matrix = {}
    for line in file:
        values = line.split()
        word = values[0]
        embedding = np.array(values[1:], dtype='float64')
        embedding_matrix[word] = embedding

    return embedding_matrix


# In[ ]:


embedding_matrix = intialize_emb_matrix(file)


# In[ ]:


def get_emb_data(data, max_len):
#     max_len = 168
    embedding_data = np.zeros((len(data), max_len, 50))  # from glove6B50d

    for idx in range(data.shape[0]):
        words_in_sentence = data[idx].split()

        for i in range(len(words_in_sentence)):
            if embedding_matrix.get(words_in_sentence[i].lower()) is not None:
                embedding_data[idx][i] = embedding_matrix[words_in_sentence[i].lower()]

    return embedding_data


# In[ ]:


# representing each word in the sentence acc. to glove embedding [str --> numerical]
X_temb = get_emb_data(X_train, 168)


# In[ ]:


# convert outputs to categorical variables
from keras.utils.np_utils import to_categorical


# In[ ]:


# converting y_train to one hot vectors so that cross-entropy loss can be used.
y_train = to_categorical(y_train)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, Dense, Dropout


# In[ ]:


model = Sequential()


# In[ ]:


model.add(LSTM(units = 256, return_sequences=True, input_shape = (168,50)))
model.add(Dropout(0.3))
model.add(LSTM(units=128))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=20, activation='softmax'))


# In[ ]:


model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['acc'])


# In[ ]:


# model Training
res = model.fit(X_temb, y_train, validation_split=0.2, batch_size=32, epochs=10, verbose=2)


# In[ ]:


test = pd.read_csv("Test.csv")
X_test = test['TEXT'].values


# In[ ]:


X_temb_test = get_emb_data(X_test, 168)


# In[ ]:


y_pred = model.predict(X_temb_test)


# In[ ]:


y_pred_emojis = []
for pred in y_pred:
    label = np.argmax(pred)
    emoji = mapping_dict[label]
    y_pred_emojis.append(emoji)


# In[ ]:





# In[ ]:


import emoji
import numpy as np
import pandas as pd

mapping = pd.read_csv("Mapping.csv")
output = pd.read_csv("OutputFormat.csv")
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

mapping = mapping.drop("Unnamed: 0", axis=1)
mapping = mapping.set_index('number')
mapping_dict = mapping.to_dict()
mapping_dict = mapping_dict['emoticons']

train = train.drop('Unnamed: 0', axis=1)
X_train = train['TEXT'].values
y_train = train['Label'].values

file = open("glove.6B.50d.txt", encoding = 'utf8')

def intialize_emb_matrix(file):
    embedding_matrix = {}
    for line in file:
        values = line.split()
        word = values[0]
        embedding = np.array(values[1:], dtype='float64')
        embedding_matrix[word] = embedding

    return embedding_matrix

embedding_matrix = intialize_emb_matrix(file)

def get_emb_data(data, max_len):
    embedding_data = np.zeros((len(data), max_len, 50))

    for idx in range(data.shape[0]):
        words_in_sentence = data[idx].split()

        for i in range(len(words_in_sentence)):
            if embedding_matrix.get(words_in_sentence[i].lower()) is not None:
                embedding_data[idx][i] = embedding_matrix[words_in_sentence[i].lower()]

    return embedding_data

X_temb = get_emb_data(X_train, 168)

y_train = pd.get_dummies(y_train).values

from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, Dense, Dropout

model = Sequential()
model.add(LSTM(units = 256, return_sequences=True, input_shape = (168,50)))
model.add(Dropout(0.3))
model.add(LSTM(units=128))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=20, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

res = model.fit(X_temb, y_train, validation_split=0.2, batch_size=32, epochs=10, verbose=2)



# In[ ]:


# Get user input
text = input(" ")

# Preprocess the text
X_test = np.array([text])
X_temb_test = get_emb_data(X_test, 168)

# Predict the label
y_pred = model.predict(X_temb_test)[0]
label = np.argmax(y_pred)
emoji = mapping_dict[label]

# Print the predicted emoji
print("Predicted Emoji: {}".format(emoji))


# In[ ]:





# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/nlp project')


# In[ ]:


pip install transformers


# In[ ]:


pip install emoji


# In[ ]:


import emoji
import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

# Load train and test data
train_data = pd.read_csv("train_emoji.csv", header=None)
test_data = pd.read_csv("test_emoji.csv", header=None)
train_data.drop(labels=[2, 3], axis=1, inplace=True)

# Map labels to corresponding emojis
emoji_mapping = {
    '0': ':beating_heart:',
    '1': ':baseball:',
    '2': ':beaming_face_with_smiling_eyes:',
    '3': ':angry_face:',
    '4': ':face_savoring_food:'
}
for key, value in emoji_mapping.items():
    emoji_mapping[key] = emoji.emojize(value)

# Tokenize sentences using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 64

X_train = train_data[0].values
y_train = train_data[1].values
X_test = test_data[0].values
y_test = test_data[1].values

X_train_encoded = tokenizer.batch_encode_plus(
    X_train,
    add_special_tokens=True,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_attention_mask=True,
    return_token_type_ids=False,
    return_tensors='tf'
)
X_test_encoded = tokenizer.batch_encode_plus(
    X_test,
    add_special_tokens=True,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_attention_mask=True,
    return_token_type_ids=False,
    return_tensors='tf'
)

# Convert labels to categorical variables
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)

# Load BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Extract features from BERT model
input_ids_train = X_train_encoded['input_ids']
attention_masks_train = X_train_encoded['attention_mask']
input_ids_test = X_test_encoded['input_ids']
attention_masks_test = X_test_encoded['attention_mask']

train_features = bert_model(input_ids_train, attention_mask=attention_masks_train)[0]
test_features = bert_model(input_ids_test, attention_mask=attention_masks_test)[0]

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_features,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    verbose=1
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_features, y_test)
print('Test accuracy:', test_acc)

# Print predictions and actual labels with corresponding sentences
predicted_labels = np.argmax(model.predict(test_features), axis=1)
actual_labels = np.argmax(y_test, axis=1)

for t in range(len(test_data)):
    print(test_data[0].iloc[t])
    print("Predictions: ", emoji.emojize(emoji_mapping[str(predicted_labels[t])]))
    print("Actual: ", emoji.emojize(emoji_mapping[str(actual_labels[t])]))
    print()

# Compute precision, recall, f1-score and accuracy
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1_score, _ = precision_recall_fscore_support(actual_labels, predicted_labels, average='weighted')
accuracy = (np.array(predicted_labels) == np.array(actual_labels)).mean()

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)
print("Accuracy: ", accuracy)

























import matplotlib.pyplot as plt

# Train the model
history = model.fit(
    train_features,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    verbose=1
)

# Access the loss and accuracy values from the history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the loss graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot the accuracy graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_features, y_test)
print('Test accuracy:', test_acc)







# In[ ]:


import matplotlib.pyplot as plt

# Train the model
history = model.fit(
    train_features,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    verbose=1
)

# Access the loss and accuracy values from the history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the loss graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot the accuracy graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_features, y_test)
print('Test accuracy:', test_acc)


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support

# Make predictions on test data
predicted_labels = model.predict(test_features)
predicted = np.argmax(predicted_labels, axis=1)
actual = np.argmax(y_test, axis=1)

# Calculate precision, recall, f1-score, and support for each class
precision, recall, f1_score, support = precision_recall_fscore_support(actual, predicted)

# Plot scatter plot of f1 scores
plt.figure(figsize=(8, 6))
plt.scatter(range(len(f1_score)), f1_score)
plt.xticks(range(len(f1_score)), list(emoji_mapping.values()), rotation='vertical')
plt.xlabel('Emoji')
plt.ylabel('F1 Score')
plt.title('F1 Score by Emoji Class')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Compute precision, recall, f1-score and accuracy
from sklearn.metrics import precision_recall_fscore_support

predicted_labels = np.argmax(model.predict(test_features), axis=1)
actual_labels = np.argmax(y_test, axis=1)
precision, recall, f1_score, _ = precision_recall_fscore_support(actual_labels, predicted_labels, average='weighted')
accuracy = (np.array(predicted_labels) == np.array(actual_labels)).mean()

# Plot a scatter plot of F1 scores
plt.figure(figsize=(8, 6))
plt.scatter(range(1), f1_score)
plt.xticks(range(1), [''])
plt.ylabel('F1 Score')
plt.title('F1 Score on Test Data')
plt.show()

