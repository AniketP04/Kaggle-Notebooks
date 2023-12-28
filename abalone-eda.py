#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries and Loading Data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mat
import shap
import polars as pl
import catboost
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import f_oneway
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, RepeatedKFold
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.svm  import LinearSVR
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from IPython.core.interactiveshell import InteractiveShell
from explainerdashboard import RegressionExplainer, ExplainerDashboard


# Open the file and see what data we will work with.

df = pd.read_csv('../EDA/data/abalone.csv')
df


# **Data Description**
# 
# **Signs**
# - `Sex` - M(male), F(female) and I (baby)
# - `Length` - length in mm (largest linear size of abalone)
# - `Diameter` - diameter in mm, perpendicular to the length
# - `Height` - height in mm from pulp
# - `Whole weight` - total weight of the eyelet in grams
# - `Shucked weight` - pulp weight in grams
# - `Viscera weight` - viscera weight in grams
# - `Shell weight` - shell weight in grams
# 
# **_Target feature_**
# - `Age` - age is calculated as rings(number of rings) + 1.5

# Let's see what data we have to work with.


df.head()


# Only sex is a categorical feature. The rest are numeric.
# Let's look at the sample size.


df.shape


# There are 4177 objects in the dataset. 9 columns (8 features) and a column that is responsible for the target.
# Let's look at data types and missing values.



df.info()


# Gaps in columns: Diameter, Whole weight and Shell weight.

# With describe(), let's look at the main statistical characteristics of the data.
# 



df.describe(include='all')


# - There are 4 unique values ​​in the Sex column. Most of the ears are male.
# - In numerical features, the values ​​are unique. There are no negative or very high values. The mean and median values ​​are similar.

# 
# Let's check the data for bagels.


df.duplicated().sum()


# There are no duplicates. Let's reduce the names in the columns to lower case.



df.columns = df.columns.str.lower()
df.columns


# 
# Let's look at the correlation matrices.

# * # Pearson correlation



df_num = df.drop('sex', axis=1)
corr = df_num.corr()

sns.heatmap(corr, cmap="Blues_r", annot=True)


# * # Spearman correlation



corr = df_num.corr(method='spearman')

sns.heatmap(corr, cmap="Blues_r", annot=True)


# * # Kendala Correlation



corr = df_num.corr(method='kendall')

sns.heatmap(corr, cmap="Blues_r", annot=True)


# There is no significant difference in the obtained correlation coefficients.
# The highest correlation coefficient of the target with the weight of the eyelet. And the minimum with the weight of the pulp.
# Other coefficients are close.

# Let's look at the correlation between numerical and categorical trait (gender) using ANOVA.



Data = []

for c1 in df.columns:
    for c2 in df.columns:
        if df[c1].dtype == 'object' and df[c2].dtype != 'object':
            CategoryGroupLists = df.groupby(c1)[c2].apply(list)
            AnovaResults = f_oneway(*CategoryGroupLists)

            if AnovaResults[1] >= 0.05:
                Data.append({'Category' : c1, 'Numerical' : c2, 'Is correlated' : 'No'})
            else:
                Data.append({'Category' : c1, 'Numerical' : c2, 'Is correlated' : 'Yes'})

AnovaRes = pd.DataFrame.from_dict(Data)
AnovaRes


# According to Annova's conclusion, there is a correlation between gender and physical size and gender.

# ## Rings - number of rings



def desc(col):
    print('Description')
    print(df[col].describe())
    print('Value_counts')
    print(df[col].value_counts(dropna=False))
    sns.histplot(data = df[col], kde = True)




desc('rings')


# We have a tail in the data. It is not very difficult, but can be complicated by model prediction.
# 
# You can try passing the logarithm of y instead of y to the model.



sns.histplot(data = np.log(df['rings']), kde = True)


# 
# Calculate the target according to the dataset description (age is the number of rings + 1.5): age = 1.5 + rings



df['age'] = df['rings'] + 1.5



df.head()


# ## Sex - gender


desc('sex')


# Rename f to F.



df['sex'] = df['sex'].replace({'f' : 'F'})




desc('sex')


# The number of ears male and female are approximately equal. Slightly smaller shell babies.

# 
# Let's look at the dependence of average age on the sex of shells.



plt.figure(figsize=(6,4))

sns.barplot(x='sex', y='age', data = df, palette='Blues')
plt.title('Sex - Age')
plt.show()


# The graph shows that women's ears are slightly heavier than men's. The weight of babies' ears is minimal.



num_cols = [ 'length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight']
for c in num_cols:
    sns.histplot(df[c])
    plt.show()


# Let's look at the dependence of the average weight on the length, diameter, height, total weight, pulp weight, viscera weight, shell weight.



num_cols = [ 'length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight']
for c in num_cols:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=c, y="age", hue="sex")
    plt.title(f'{c} - Age')
    plt.show()


# According to all graphs, the minimum dimensions and weights refer to baby barnacles. There is no significant difference between male and female shells.



plt.figure(figsize=(10, 8))
df[num_cols].boxplot()


# We have outliers in total weight, pulp weight, viscera weight, and shell weight.



sns.pairplot(df, hue="sex")


# Pairplot shows that the minimum dimensions and weights are for baby shells. There is no significant difference between male and female shells.

# ## Dataset upsampling

# Let's set a function that will increase the number of lines to the minimum. We pass to the input the dataframe that needs to be multiplied and the size of the dataset that we should get at the output.



def upsample(data,rows):
    data = shuffle(data, random_state=12345)
    repeat = math.ceil(rows/len(data))
    data = pd.concat([data]*repeat, ignore_index=True)
    data = shuffle(data, random_state=12345)
    data = data[:rows]
    return data




df_mln = upsample(df, 1000000)
len(df_mln)


# ## Performance comparison of Pandas vs Polars

# * Data upload



get_ipython().run_cell_magic('time', '', "#Polaris\ndf_pl = pl.read_csv('../EDA/data/abalone.csv')")




get_ipython().run_cell_magic('time', '', "# Pandas\ndf_p = pd.read_csv('../EDA/data/abalone.csv')")


# Polaris is faster with loading.

# * Filtration



get_ipython().run_cell_magic('time', '', "# Polars\ndf_pl[['Sex', 'Rings']]")




get_ipython().run_cell_magic('time', '', "# Pandas\ndf_p[['Sex', 'Rings']]")


# In[32]:


get_ipython().run_cell_magic('time', '', "# Pandas\ndf_p[(df_p.Sex == 'M') & (df_p.Rings > 10)]")


# In[33]:


get_ipython().run_cell_magic('time', '', "# Polars\ndf_pl.filter((pl.col('Sex') == 'M') & (pl.col('Rings') > 10))")


# Polaris wins again in filtering.

# 
# * Data aggregation

# In[34]:


get_ipython().run_cell_magic('time', '', "# Pandas\ndf_p.groupby('Sex').agg({'Whole weight' : 'mean', 'Diameter' : 'max'})")


# In[35]:


get_ipython().run_cell_magic('time', '', "# Polars\ndf_pl.groupby('Sex').agg([pl.mean('Whole weight'), pl.max('Diameter')])")


# In[36]:


get_ipython().run_cell_magic('time', '', "# lazy operations\nq = (\n    df_pl\n    .lazy()\n    .groupby(by='Sex')\n    .agg(\n        [\n            pl.col('Whole weight').mean(),\n            pl.col('Diameter').max()\n        ]\n    )\n)\n\nq.collect()")


# Polaris is faster at aggregation, but lazy operations are slower than aggregation in a similar way to pandas.

# ## ## Building a baseline

# * encode categorical features using TargetEncoder (we only have sex)
# * scale all features using StandardScaler
# * train linear regression

# 
# Divide the data into target y and feature matrix X.

# In[37]:


X = df.drop(['age','rings'], axis=1)
y = df['age']


# Let's look at gaps in the data.

# In[38]:


X.isna().sum()


# 
# Fill in the gaps with the median value.

# In[39]:


for c in ['diameter','whole weight','shell weight']:
    X[c] = X[c].fillna(X[c].median())


# 
# Let's check how the filling worked.

# In[40]:


X.isna().sum()


# There are no passes. We divide the data into test and training samples.

# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=911)


# 
# Let's save the categorical column sex - cat_features and the numeric columns in num_features to the list.

# In[42]:


cat_features = ['sex']
num_features = ['length', 'diameter', 'height', 'whole weight', 'shucked weight','viscera weight', 'shell weight']


# 
# We arrange the stages of working with data in the pipeline, as well as a model for training.
# We start training and predict the age of the ears for the test sample.
# We consider metrics (MAE, RMSE, MAPE).

# 
# * Run on numerical signs.

# In[43]:


p1_num = Pipeline([
    ('scaler_', StandardScaler()),
    ('model_', LinearRegression())
    ])
p1_num.fit(X_train[num_features], y_train)
pred = p1_num.predict(X_test[num_features])
score_mae_p1_num = mean_absolute_error(y_test, pred)
print("The Mean Absolute Error of our Model is {}".format(round(score_mae_p1_num, 2)))
score_rmse_p1_num = np.sqrt(mean_absolute_error(y_test, pred))
print("The Root Mean Squared Error of our Model is {}".format(round(score_rmse_p1_num, 2)))
MAPE_p1_num = mean_absolute_percentage_error (y_test, pred)
print("The Mean Absolute Percentage Error of our Model is {}".format(round(MAPE_p1_num, 2)))


# * Trigger on a numeric + categorical feature.
# 

# In[44]:


p1 = Pipeline([
    ('encoder_',TargetEncoder(cols=cat_features)),
    ('scaler_', StandardScaler()),
    ('model_', LinearRegression())
    ])
p1.fit(X_train, y_train)
pred = p1.predict(X_test)
score_mae_p1 = mean_absolute_error(y_test, pred)
print("MAE {}".format(round(score_mae_p1, 2)))
score_rmse_p1 = np.sqrt(mean_absolute_error(y_test, pred))
print("RMSE {}".format(round(score_rmse_p1, 2)))
MAPE_p1 = mean_absolute_percentage_error (y_test, pred)
print("MAPE {}".format(round(MAPE_p1, 2)))


# 
# **Conclusion on adding a categorical feature:** When adding the categorical feature sex, the metrics improved.

# Let's try instead of * Run on numerical features. y apply to the logarithm of y, get rid of the tail in the distribution.
# Let's train the model on the same pipeline and compare the resulting metrics.

# * Run on numerical signs.

# In[45]:


y_train_l = np.log(y_train)
y_test_l = np.log(y_test)
p1_num.fit(X_train[num_features], y_train_l)
pred = p1_num.predict(X_test[num_features])
score_mae_p1_l_num = mean_absolute_error(np.exp(y_test_l), np.exp(pred))
print("MAE {}".format(round(score_mae_p1_l_num, 2)))
score_rmse_p1_l_num = np.sqrt(mean_absolute_error(np.exp(y_test_l), np.exp(pred)))
print("RMSE {}".format(round(score_rmse_p1_l_num, 2)))
MAPE_p1_l_num = mean_absolute_percentage_error(np.exp(y_test_l),np.exp(pred))
print("MAPE {}".format(round(MAPE_p1_l_num, 2)))


# 
# * Trigger on a numeric + categorical feature.

# In[46]:


p1.fit(X_train, y_train_l)
pred = p1.predict(X_test)
score_mae_p1_l = mean_absolute_error(np.exp(y_test_l), np.exp(pred))
print("MAE {}".format(round(score_mae_p1_l, 2)))
score_rmse_p1_l = np.sqrt(mean_absolute_error(np.exp(y_test_l), np.exp(pred)))
print("RMSE {}".format(round(score_rmse_p1_l, 2)))
MAPE_p1_l = mean_absolute_percentage_error(np.exp(y_test_l),np.exp(pred))
print("MAPE {}".format(round(MAPE_p1_l, 2)))


# **Conclusion on the logarithm of the target:** The metrics of the model deteriorated, as when training was started only on numerical features. So it is with the addition of categorical.

# 
# And now we will try different models to choose the best one, which we will adjust the exact parameters of the model. We will run on all signs (categorical + numerical).

# * Run on numerical signs.

# In[47]:


list_pipelines_ = []
mean_score_ = []
models = [
    KNeighborsRegressor(),
    LinearSVR(random_state=911),
    DecisionTreeRegressor(random_state=911),
    RandomForestRegressor(random_state=911),
    AdaBoostRegressor(random_state=911),
    GradientBoostingRegressor(random_state=911),
    CatBoostRegressor(random_seed=911,  verbose=False),
    lgb.LGBMRegressor(random_state=911),
    xgb.XGBRegressor(seed=911)
    ]
scalers = [StandardScaler()]

for m in models:
    for scaler in scalers:
        p2 = Pipeline([
             ('scaler_', scaler),
              ('model_', m)
              ])
        scores_ = cross_val_score(p2, X[num_features], y)
        print(scaler)
        print(m)
        print("model score: %.3f" % scores_.mean())
        print()
        mean_score_.append(scores_.mean())
        list_pipelines_.append(str(m) + str(scaler))


# 
# Let's look at the results. Let's build a graph and write them into a dataframe.

# In[48]:


results_num=pd.DataFrame(data=mean_score_,index=list_pipelines_)
results_num.plot(figsize=(15,7))
plt.grid()
results_num


# In[49]:


max_value = max(mean_score_)
max_index = mean_score_. index(max_value)
results_num.iloc[max_index]


# **Fitting inference for numeric data** The best score of 0.450406 was given to the CatBoostRegressor model.

# 
# * Trigger on a numeric + categorical feature.

# In[50]:


list_pipelines_ = []
mean_score_ = []
models = [
    KNeighborsRegressor(),
    LinearSVR(random_state=911),
    DecisionTreeRegressor(random_state=911),
    RandomForestRegressor(random_state=911),
    AdaBoostRegressor(random_state=911),
    GradientBoostingRegressor(random_state=911),
    CatBoostRegressor(random_seed=911,  verbose=False),
    lgb.LGBMRegressor(random_state=911),
    xgb.XGBRegressor(seed=911)
    ]
scalers = [StandardScaler()]

for m in models:
    for scaler in scalers:
        p2 = Pipeline([
            ('encoder_',TargetEncoder(cols=cat_features)),
             ('scaler_', scaler),
              ('model_', m)
              ])
        scores_ = cross_val_score(p2, X, y)
        print(scaler)
        print(m)
        print("model score: %.3f" % scores_.mean())
        print()
        mean_score_.append(scores_.mean())
        list_pipelines_.append(str(m) + str(scaler))


# 
# Let's look at the results. Let's build a graph and write them into a dataframe.

# In[51]:


results=pd.DataFrame(data=mean_score_,index=list_pipelines_)
results.plot(figsize=(15,7))
plt.grid()
results


# In[52]:


max_value = max(mean_score_)
max_index = mean_score_. index(max_value)
results.iloc[max_index]


# 
# **Inference by fitting the model (numerical + categorical feature) ** best score increased and amounted to 0.463876. GradientBoostingRegressor model.

# 
# Let's look at the metrics for now without the selection of hyper-parameters.
# * Run on numerical signs. CatBoostRegressor model.

# In[53]:


p2_num = Pipeline([
     ('scaler_', StandardScaler()),
      ('model', CatBoostRegressor(random_seed=911,  verbose=False))
       ])
p2_num.fit(X_train[num_features], y_train)
y_pred = p2_num.predict(X_test[num_features])
score_mae_p2_num = mean_absolute_error(y_test, y_pred)
print("MAE {}".format(round(score_mae_p2_num, 2)))
score_rmse_p2_num = np.sqrt(mean_absolute_error(y_test, y_pred))
print("RMSE {}".format(round(score_rmse_p2_num, 2)))
MAPE_p2_num = mean_absolute_percentage_error (y_test, y_pred)
print("MAPE {}".format(round(MAPE_p2_num, 2)))


# * Trigger on a numeric + categorical feature. GradientBoostingRegressor.

# In[54]:


p2 = Pipeline([
    ('encoder_',TargetEncoder(cols=cat_features)),
     ('scaler_', StandardScaler()),
      ('model', GradientBoostingRegressor(random_state=911))
       ])
p2.fit(X_train, y_train)
y_pred = p2.predict(X_test)
score_mae_p2 = mean_absolute_error(y_test, y_pred)
print("MAE{}".format(round(score_mae_p2, 2)))
score_rmse_p2 = np.sqrt(mean_absolute_error(y_test, y_pred))
print("RMSE {}".format(round(score_rmse_p2, 2)))
MAPE_p2 = mean_absolute_percentage_error (y_test, y_pred)
print("MAPE {}".format(round(MAPE_p2, 2)))


# **Choice by model selection only for numerical and with categorical features** different models are selected. But the quality of the best model with a categorical feature is higher than without it.

# * Run on numerical signs. LinearRegression and CatBoostRegressor models.

# In[55]:


print(f'MAE {round(score_mae_p1_num,2), round(score_mae_p2_num,2)}')
print(f"RMSE {round(score_rmse_p1_num,2), round(score_rmse_p2_num,2)}")
print(f"MAPE {round(MAPE_p1_num,2), round(MAPE_p2_num,2)}")


# 
# **Running with a categorical feature:** Compare the metrics obtained with LinearRegression and GradientBoostingRegressor.

# 

# In[56]:


print(f'MAE {round(score_mae_p1,2), round(score_mae_p2,2)}')
print(f"RMSE{round(score_rmse_p1,2),round(score_rmse_p2,2)}")
print(f"MAPE {round(MAPE_p1,2),round(MAPE_p2,2)}")


# 
# **Best Model Run Conclusion:**Runtime errors on all features have decreased. Fitted models provide better metrics than LinearRegression.

# 
# Let's make a selection of hyper-parameters of the models.

# In[57]:


p3_num = Pipeline([
     ('scaler_', StandardScaler()),
      ('model', GradientBoostingRegressor())
       ])
param_grid = {'model__n_estimators' : [10, 50, 100, 500, 1000],
              'model__learning_rate':  [ 0.01, 0.05, 0.07],
              'model__subsample': [0.5, 0.7, 1.0],
              'model__max_depth': [3, 7, 9]
              }
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = GridSearchCV(p3_num, param_grid=param_grid, n_jobs=-1, cv=cv)
grid.fit(X_train[num_features], y_train)
model = grid.best_estimator_
y_pred = model.predict(X_test[num_features])


# In[58]:


grid.best_params_


# In[59]:


score_mae_p3_num = mean_absolute_error(y_test, y_pred)
print("MAE {}".format(round(score_mae_p3_num, 2)))
score_rmse_p3_num = np.sqrt(mean_absolute_error(y_test, y_pred))
print("RMSE {}".format(round(score_rmse_p3_num, 2)))
MAPE_p3_num = mean_absolute_percentage_error (y_test, y_pred)
print("MAPE {}".format(round(MAPE_p3_num, 2)))


# In[60]:


from sklearn.model_selection import RepeatedKFold
p4 = Pipeline([
    ('encoder_',TargetEncoder(cols=cat_features)),
     ('scaler_', StandardScaler()),
      ('model', GradientBoostingRegressor())
       ])
param_grid = {'model__n_estimators' : [10, 50, 100, 500, 1000],
              'model__learning_rate':  [ 0.01, 0.05, 0.07],
              'model__subsample': [0.5, 0.7, 1.0],
              'model__max_depth': [3, 7, 9]
              }
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = GridSearchCV(p4, param_grid=param_grid, n_jobs=-1, cv=cv)
grid.fit(X_train, y_train)
model = grid.best_estimator_
y_pred = model.predict(X_test)


# In[61]:


grid.best_params_


# In[62]:


score_mae_p4 = mean_absolute_error(y_test, y_pred)
print("МАЕ {}".format(round(score_mae_p4, 2)))
score_rmse_p4 = np.sqrt(mean_absolute_error(y_test, y_pred))
print("RMSE {}".format(round(score_rmse_p4, 2)))
MAPE_p4 = mean_absolute_percentage_error (y_test, y_pred)
print("MAPE {}".format(round(MAPE_p4, 2)))


# 
# Let's compare the errors in learning only on numerical features.

# In[63]:


print(f'MAE {score_mae_p1_num, score_mae_p2_num, score_mae_p3_num}')
print(f"RMSE {score_rmse_p1_num,score_rmse_p2_num,score_rmse_p3_num}")
print(f"MAPE {MAPE_p1_num,MAPE_p2_num, MAPE_p3_num}")


# **Conclusion** The error gradually decreases.

# 
# Let's look at the errors on all features (categorical + numerical):

# In[64]:


print(f'The Mean Absolute Error of our Model {score_mae_p1, score_mae_p2, score_mae_p4}')
print(f"The Root Mean Squared Error of our Model is {score_rmse_p1,score_rmse_p2,score_rmse_p4}")
print(f"The Mean Absolute Percentage Error of our Model is {MAPE_p1,MAPE_p2, MAPE_p4}")


# 
# **Conclusion** The error is gradually decreasing.

# 
# Let's look at the average value for the target.

# In[65]:


np.mean(y)


# Let's compare the error with the average for the target.

# In[66]:


mean_absolute_error(y_test, y_pred)/y.mean()


# In[67]:


sns.histplot(df['age'])


# 
# Let's look at MAE with an age of more than 20 years and less than 5 years. Compare them.

# In[68]:


print(mean_absolute_error(y_test[y_test>20], y_pred[y_test>20]))
mean_absolute_percentage_error(y_test[y_test>20], y_pred[y_test>20])


# In[69]:


print(mean_absolute_error(y_test[y_test<5], y_pred[y_test<5]))
mean_absolute_percentage_error(y_test[y_test<5], y_pred[y_test<5])


# Model error inference: According to MAPA, the percentage of error on ears over 20 years of age is lower than on ears under 5 years of age. At the same time, MAE is, of course, higher for data with a target of more than 20 years.

# 
# ### Shaple values

# Let's look at how the model works using the shapley library's shapley method.
# Our model:

# In[81]:


grid.best_estimator_


# Let's prepare the values. Let's do the coding and scaling.

# In[82]:


explainer = shap.TreeExplainer(grid.best_estimator_['model'])
observations = grid.best_estimator_['encoder_'].transform(X)
observations = grid.best_estimator_['scaler_'].transform(observations)
shap_values = explainer.shap_values(observations)


# Let's build a summary_plot to evaluate which features the model considers the most important.

# In[83]:


shap.summary_plot(shap_values, X, plot_type="bar")


# The most important feature is the weight of the shell. Slightly less important is the weight of the flesh. And the total weight is only on the 3rd place in importance.
# The least important features, with approximately identical meanings, are sex, diameter and length.
# A little more important is the height of the ear and the weight of the insides.
# 
# It turns out that weight is a more important feature for the model than the geometric dimensions of the mollusk.

# Let's look at the direction in which the forecast of the model shifts based on the features of the eyelet.

# In[84]:


shap.summary_plot(shap_values, X)


# The graph shows that only 2 features have an inverse relationship with the target. This is the length and weight of the viscera. The higher this value, the lower the age. Although, the length is not so clear.
# The remaining features with increasing values ​​lead to an increase in the target.
# It turns out that an increase in the total weight, shell weight, diameter and total weight of the mollusk is interpreted by the model as an increase in age.
# The smaller these parameters, the greater the age of the object.
