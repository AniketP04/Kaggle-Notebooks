#!/usr/bin/env python
# coding: utf-8

# # Optiver Trading at the Close

# ## Import Libraries

# In[1]:


import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
temp = dict(layout = go.Layout(font = dict(family="Franklin Gothic", size=12), width = 1500))

import torch
from torch import nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset, random_split
import time


# ## Importing Dataset

# In[2]:


df = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


def inspect_columns(df):
    
    result = pd.DataFrame({
        'unique': df.nunique() == len(df),
        'cardinality': df.nunique(),
        'with_null': df.isna().any(),
        'null_pct': round((df.isnull().sum() / len(df)) * 100, 2),
        '1st_row': df.iloc[0],
        'random_row': df.iloc[np.random.randint(low=0, high=len(df))],
        'last_row': df.iloc[-1],
        'dtype': df.dtypes
    })
    return result

inspect_columns(df)


# # EDA

# In[6]:


plt.figure(figsize=(12,5))
plt.title("Distribution of Target")
ax = sns.distplot(df['target'])


# In[7]:


plt.figure(figsize=(30,30))
corr=df.corr()
sns.heatmap(corr,annot=True,cmap='mako',mask=np.triu(corr))
plt.show()


# In[8]:


stock_id_0_df = df[df['stock_id'] == 0].head(700) 
stock_id_0_df


# In[9]:


fig = go.Figure()

fig.add_trace(
    go.Scatter(x = stock_id_0_df['time_id'], 
            y = stock_id_0_df['ask_price'], 
            name = 'ask price',
            line = dict(color = 'blue')))

fig.add_trace(
    go.Scatter(x = stock_id_0_df['time_id'], 
            y = stock_id_0_df['bid_price'], 
            name = 'bid price',
            line = dict(color = 'green')))


fig.update_layout(title = "Overview for Ask Price and Bid Price",
                title_font = dict(size = 15),
                showlegend = True,
                width = 1000,
                height = 400,
                margin = dict(l = 40, r = 40, t = 40, b = 20))


# In[10]:


fig = go.Figure()

fig.add_trace(
    go.Scatter(x = stock_id_0_df['time_id'], 
            y = stock_id_0_df['target'], 
            name = 'Target variable',
            line = dict(color = 'red'),
            yaxis = "y2"))

fig.add_trace(
    go.Scatter(x = stock_id_0_df['time_id'], 
            y = stock_id_0_df['wap'], 
            name = 'WAP',
            line = dict(color = 'black')))

fig.update_layout(title = "Overview for WAP and Target Variable",
                title_font = dict(size = 15),
                showlegend = True,
                width = 1000,
                height = 400,
                margin = dict(l = 40, r = 40, t = 40, b = 20),
                yaxis2 = dict(title = "Target Variable", overlaying = "y", side = "right"),)


# # Model: LSTM with Attention 

# In[11]:


def add_historic_features(df, cols, shifts=3, add_first=True):
    for col in cols:
        grouped_vals = df[["stock_id", "date_id", col]].groupby(["stock_id","date_id"])
        fill_value = df[col].mean()
        
        for shift in np.arange(shifts):
            df[col+"_shift"+str(shift+1)] = grouped_vals.shift(shift+1).fillna(fill_value)
    
        if add_first:
            df = df.merge(grouped_vals.first().reset_index(), on=["date_id","stock_id"], suffixes=["","_first"])
    return df


# In[12]:


def fillmean(df, cols):
    for col in cols:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
    return df


# In[13]:


def add_info_columns(raw_df):
    df = raw_df.copy()
    
    df[["reference_price", "far_price","near_price","bid_price","ask_price","wap"]] = df[["reference_price", "far_price","near_price","bid_price","ask_price","wap"]].fillna(1.0)
    df = fillmean(df, ["imbalance_size", "matched_size"])
    
    df['imbalance_ratio'] = df['imbalance_size'] / (df['matched_size'] + 1.0e-8)
    df["imbalance"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
    
    df['ordersize_imbalance'] = (df['bid_size']-df['ask_size']) / ((df['bid_size']+df['ask_size'])+1.0e-8)
    df['matching_imbalance'] = (df['imbalance_size']-df['matched_size']) / ((df['imbalance_size']+df['matched_size'])+1.0e-8)
    
    df = add_historic_features(df, ["imbalance","imbalance_ratio","reference_price","wap","matched_size","far_price","near_price"], shifts=6, add_first=True)
    
    return df


# In[14]:


df = add_info_columns(df)

nullsum = df.isna().sum(axis=0)
print(nullsum[nullsum != 0])

df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)
print(df)


# In[15]:


x_cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id', 'target']]
y_cols = ["target"]


# In[16]:


means = df[x_cols].mean(0)
stds = df[x_cols].std(0)


# In[17]:


def normalize_features(x):
    return (x - means) / (stds+1e-8)

def get_xy(df):
    x = df[x_cols]
    x = normalize_features(x)
    
    y = df[y_cols]
    
    return x.values, y.values


# In[18]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[19]:


def get_dataloaders(df, batch_size=512):
    (x,y) = get_xy(df)
    
    x_tensor = torch.Tensor(x).to(device)
    y_tensor = torch.Tensor(y).to(device)
    
    full_dataset = TensorDataset(x_tensor, y_tensor)
    train_dataset, test_dataset = random_split(full_dataset, [0.8,0.2])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=min(batch_size*4, len(test_dataset)), drop_last=True)
    return (train_dataloader, test_dataloader)

train_dataloader, test_dataloader = get_dataloaders(df)


# In[20]:


class LSTMwithAttention(nn.Module):
    def __init__(self, input_size=66, hidden_layer_size=50, output_size=1, num_layers=5):
        super().__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.batch_size = 1   

        self.conv1d = nn.Conv1d(in_channels=66, out_channels=20, kernel_size=2)
        self.lstm = nn.LSTM(20, hidden_layer_size, num_layers, dropout=0.2)

        layers = [
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.Linear(hidden_layer_size, output_size)
                ]

        self.linear_layers = nn.Sequential(*layers)

        init_rnn(self.lstm, 'xavier')
        self.dropout = nn.Dropout(0.5)
        self.attn_weights = None

    def attention_net(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden = final_state.view(-1, self.hidden_layer_size, self.num_layers)
        self.attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)

        soft_attn_weights = torch.tanh(self.attn_weights)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(2)
        return context 

    def forward(self, input_seq):
        input_seq = input_seq.unsqueeze(0).permute(0, 2, 1)
        input_seq = self.conv1d(input_seq)
        input_seq = input_seq.permute(0, 2, 1)

        lstm_out, self.hidden_cell = self.lstm(input_seq.reshape(len(input_seq[0]), 1, 20))
        attn_output = self.attention_net(lstm_out, self.hidden_cell[0])
        
        out = self.linear_layers(attn_output.view(-1, self.hidden_layer_size))
        out = torch.tanh(out)

        return out[-1]

def init_rnn(x, type='uniform'):
    for layer in x._all_weights:
        for w in layer:
            if 'weight' in w:
                if type == 'xavier':
                    init.xavier_normal_(getattr(x, w))
                elif type == 'uniform':
                    stdv = 1.0 / (getattr(x, w).size(-1))**0.5
                    init.uniform_(getattr(x, w), -stdv, stdv)
                elif type == 'normal':
                    stdv = 1.0 / (getattr(x, w).size(-1))**0.5
                    init.normal_(getattr(x, w), 0.0, stdv)
                else:
                    raise ValueError


# In[21]:


model = LSTMwithAttention().to(device)


# In[22]:


def train_loop(dataloader, model, loss_fn, optimizer, shortcut=0):
    size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)

    train_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        
        pred = model(X)
        
#         print(pred.detach().numpy().flatten())
        
        loss = loss_fn(pred, y)
        train_loss += loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if pred.std() < 0.000001:
            print("WARNING: std() is zero, stopping")
            break
        
        if shortcut > 0 and batch == shortcut:
            return train_loss.detach().cpu().numpy() / shortcut
    return train_loss.detach().cpu().numpy() / num_batches


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
      
            
            pred = model(X)
            test_loss += loss_fn(pred, y).detach().cpu().numpy()
    
        scheduler.step(test_loss)
    return test_loss / num_batches
        
def predict(X, model):
    model.eval()
    with torch.no_grad():
        pred = model(X)
    return pred.detach().cpu().numpy().flatten()


# In[23]:


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.00001):
        self.best_model = None
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        
    def get_best_model(self):
        return self.best_model

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            print(f"New best loss: {validation_loss:>4f}")
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model = model
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# In[24]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5, verbose=True)
early_stopper = EarlyStopper(patience=15, min_delta=0.0001)

loss_fn = nn.L1Loss()


# In[25]:


history = pd.DataFrame([], columns=["epoch","train_loss","test_loss","lr"])

t1 = time.time()

for epoch in range(20):
    
    print(f"Epoch {epoch+1:>3d}",end=" ")
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, shortcut=0)
    print(f"Train: {train_loss:>5f}", end=" ")
    test_loss = test_loop(test_dataloader, model, loss_fn)
    print(f"| Test: {test_loss:>5f}")

    if early_stopper.early_stop(test_loss, model) or time.time() - t1 > 60*60*8:  
        model = early_stopper.get_best_model()
        break

    history.loc[len(history),:] = [epoch+1, train_loss, test_loss, optimizer.param_groups[0]['lr']]
history

