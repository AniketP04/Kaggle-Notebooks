#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import sys
import json
import numpy as np
from scipy import stats
from utilities import Hyperparameters
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt


# In[33]:


SCRIPT_DIR_PATH = r"X:\Project\ETH\EthereumPriceNet-master"
DATA_DIR_REL_PATH = 'data/'
RAW_DATA_DIR_REL_PATH = 'data/raw/'
DATA_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, DATA_DIR_REL_PATH)
RAW_DATA_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, RAW_DATA_DIR_REL_PATH)


# In[39]:



RESULTS_DIR_REL_PATH = 'results/' #+ datetime.now().isoformat(' ', 'seconds') + '/'

RESULTS_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, RESULTS_DIR_REL_PATH)


# In[22]:


def load_raw_data():
    f_daily_avg_gas_limit = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailyavggaslimit.json'), 'r')
    f_daily_avg_gas_price = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailyavggasprice.json'), 'r')
    f_daily_gas_used = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailygasused.json'), 'r')
    f_daily_txn_fee = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailytxnfee.json'), 'r')
    f_eth_daily_market_cap = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'ethdailymarketcap.json'), 'r')
    f_eth_daily_price = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'ethdailyprice.json'), 'r')

    json_daily_avg_gas_limit = json.load(f_daily_avg_gas_limit)
    json_daily_avg_gas_price = json.load(f_daily_avg_gas_price)
    json_daily_gas_used = json.load(f_daily_gas_used)
    json_daily_txn_fee = json.load(f_daily_txn_fee)
    json_eth_daily_market_cap = json.load(f_eth_daily_market_cap)
    json_eth_daily_price = json.load(f_eth_daily_price)

    f_daily_avg_gas_limit.close()
    f_daily_avg_gas_price.close()
    f_daily_gas_used.close()
    f_daily_txn_fee.close()
    f_eth_daily_market_cap.close()
    f_eth_daily_price.close()

    daily_avg_gas_limit_data = np.asarray([[d['unixTimeStamp'], d['gasLimit']] for d in json_daily_avg_gas_limit['result']][8:], dtype=np.float64)
    daily_avg_gas_price_data = np.asarray([[d['unixTimeStamp'], d['avgGasPrice_Wei']] for d in json_daily_avg_gas_price['result']][8:], dtype=np.float64)
    daily_gas_used_data = np.asarray([[d['unixTimeStamp'], d['gasUsed']] for d in json_daily_gas_used['result']][8:], dtype=np.float64)
    daily_txn_fee_data = np.asarray([[d['unixTimeStamp'], d['transactionFee_Eth']] for d in json_daily_txn_fee['result']][8:], dtype=np.float64)
    eth_daily_market_cap_data = np.asarray([[d['unixTimeStamp'], d['marketCap']] for d in json_eth_daily_market_cap['result']][8:], dtype=np.float64)
    eth_daily_price_data = np.asarray([[d['unixTimeStamp'], d['value']] for d in json_eth_daily_price['result']][8:], dtype=np.float64)

    print(daily_avg_gas_limit_data.shape)

    return (daily_avg_gas_limit_data[:, 1], daily_avg_gas_price_data[:, 1], daily_gas_used_data[:, 1], daily_txn_fee_data[:, 1], eth_daily_market_cap_data[:, 1], eth_daily_price_data[:, 1])


# In[23]:


def get_price_ffts(hps, eth_daily_price_data):
    windows = []
    for i in range(0, eth_daily_price_data.shape[0] - hps.fft_window_size + 1):
        window = eth_daily_price_data[i:i + hps.fft_window_size]
        windows += [stats.zscore(window)]
    windows = np.vstack(windows)
    return np.abs(np.fft.fft(windows))


# In[24]:


def get_preprocessed_data(hps, full_sequence):
    windows, y = [], []
    for i in range(0, full_sequence.shape[0] - hps.sequence_length + 1 - hps.prediction_window_size):
        window = full_sequence[i:i + hps.sequence_length, :]
        windows += [window]
        prediction_window = full_sequence[i + hps.sequence_length:i + hps.sequence_length + hps.prediction_window_size, 5]
        y += [[
            100*(np.amin(prediction_window)/window[-1, 5]-1),
            100*(np.mean(prediction_window)/window[-1, 5]-1),
            100*(np.amax(prediction_window)/window[-1, 5]-1)
        ]]
    return (np.stack(windows), np.array(y))


# In[25]:


hps = Hyperparameters()

raw_data = load_raw_data()
eth_daily_price_data = raw_data[-1]
price_ffts = get_price_ffts(hps, eth_daily_price_data)
full_sequence = np.concatenate((np.stack(raw_data, axis=1)[hps.fft_window_size - 1:, :], price_ffts), axis=1)
X, y = get_preprocessed_data(hps, full_sequence)

print('X.shape = ', X.shape)
print('y.shape = ', y.shape)

np.save(os.path.join(DATA_DIR_ABS_PATH, 'X.npy'), X)
np.save(os.path.join(DATA_DIR_ABS_PATH, 'y.npy'), y)


# In[26]:


def load_data():
    X = np.load(os.path.join(DATA_DIR_ABS_PATH, 'X.npy'))
    y = np.load(os.path.join(DATA_DIR_ABS_PATH, 'y.npy'))
    return (X, y)


# In[27]:


def generate_model(hps):
    inputs = tf.keras.Input(shape=(hps.sequence_length, 6 + hps.fft_window_size), name='lstm_inputs')
    batch_norm = tf.keras.layers.BatchNormalization()(inputs)
    lstm1 = tf.keras.layers.LSTM(units=hps.lstm1_units, return_sequences=True, kernel_regularizer=hps.lstm1_regularizer)(batch_norm)
    lstm2 = tf.keras.layers.LSTM(units=hps.lstm2_units, kernel_regularizer=hps.lstm2_regularizer)(lstm1)
    dense = tf.keras.layers.Dense(units=hps.dense_units, activation='tanh', kernel_regularizer=hps.dense_regularizer)(lstm2)
    outputs = tf.keras.layers.Dense(units=3)(dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=hps.learning_rate), metrics=['mae'])
    return model


# In[28]:


def draw_results(y, predictions, title):
    plt.figure()
    plt.plot(y[:, 0], label='actual')
    plt.plot(predictions[:, 0], label='predicted')
    plt.xlabel('day')
    plt.ylabel('%% change in price')
    plt.title(title + ' (min)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, title + '_min.png'), dpi=600, format='png')

    plt.figure()
    plt.plot(y[:, 1], label='actual')
    plt.plot(predictions[:, 1], label='predicted')
    plt.xlabel('day')
    plt.ylabel('%% change in price')
    plt.title(title + ' (avg)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, title + '_avg.png'), dpi=600, format='png')

    plt.figure()
    plt.plot(y[:, 2], label='actual')
    plt.plot(predictions[:, 2], label='predicted')
    plt.xlabel('day')
    plt.ylabel('%% change in price')
    plt.title(title + ' (max)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, title + '_max.png'), dpi=600, format='png')










    


# In[29]:


def draw_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (mse)')
    plt.title('loss')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, 'loss.png'), dpi=600, format='png')

    plt.figure()
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (mse)')
    plt.title('val loss')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_ABS_PATH, 'val_loss.png'), dpi=600, format='png')


# In[30]:


def split_data(hps, X, y):
    index = int(hps.train_split * X.shape[0])
    return (X[:index], y[:index], X[index:], y[index:])


# In[ ]:



    
with open(os.path.join(RESULTS_DIR_ABS_PATH, 'output.txt'), 'w') as f:
    X, y = load_data()
    X_train, y_train, X_val, y_val = split_data(hps, X, y)

    model = generate_model(hps)
    tf.keras.utils.plot_model(model, os.path.join(RESULTS_DIR_ABS_PATH, 'model.png'), show_shapes=True)
    model.summary()

    #if (len(sys.argv) > 1):
        #model.load_weights(sys.argv[1])

    history = model.fit(
        X_train,
        y_train,
        epochs=hps.epochs,
        batch_size=hps.batch_size,
        validation_data=(X_val, y_val)
    )

    model.save_weights(os.path.join(RESULTS_DIR_ABS_PATH, 'weights.h5'))

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    draw_results(y_train, train_predictions, 'train')
    draw_results(y_val, val_predictions, 'validation')
    draw_history(history)

    f.write('loss: {}\nmae: {}\nval loss: {}\nval mae: {}\n'.format(
        history.history['loss'][-1],
        history.history['mae'][-1],
        history.history['val_loss'][-1],
        history.history['val_mae'][-1]
    ))
    f.write('val rho (min): {}\nval rho (avg): {}\nval rho (max): {}'.format(
        np.corrcoef(y_val[:, 0], val_predictions[:, 0])[0, 1],
        np.corrcoef(y_val[:, 1], val_predictions[:, 1])[0, 1],
        np.corrcoef(y_val[:, 2], val_predictions[:, 2])[0, 1]
    ))


# **Refer this [repo](https://github.com/AniketP04/ETH-Price-Prediction/blob/main/) for the other files related to this project.**
