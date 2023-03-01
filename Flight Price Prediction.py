#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_excel('Data_Train.xlsx')


# ## EXPLORATORY DATA ANALYSIS

# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data=data.dropna()
print(data.shape)


# In[8]:


data.duplicated().sum()


# In[9]:


data = data.drop_duplicates()
print(data.shape)


# In[10]:


data.describe()


# In[11]:


print(data['Airline'].unique())
print(data['Airline'].nunique())


# In[12]:


print(data['Date_of_Journey'].unique())
print(data['Date_of_Journey'].nunique())


# In[13]:


print(data['Source'].unique())
print(data['Source'].nunique())


# In[14]:


print(data['Destination'].unique())
print(data['Destination'].nunique())


# In[15]:


print(data['Route'].unique())
print(data['Route'].nunique())


# In[16]:


print(data['Dep_Time'].unique())
print(data['Dep_Time'].nunique())


# In[17]:


print(data['Arrival_Time'].unique())
print(data['Arrival_Time'].nunique())


# In[18]:


print(data['Duration'].unique())
print(data['Duration'].nunique())


# In[19]:


print(data['Total_Stops'].unique())
print(data['Total_Stops'].nunique())


# In[20]:


print(data['Additional_Info'].unique())
print(data['Additional_Info'].nunique())


# In[21]:


print(data['Price'].unique())
print(data['Price'].nunique())


# ### Count Plots

# In[22]:


data.pivot_table(data[['Price']],index=['Airline'],aggfunc='sum').plot(kind='bar',figsize=(10,5)) 
plt.xlabel('Airline')
plt.ylabel('Price')


# In[23]:


data['Airline'].value_counts()


# In[24]:


data.pivot_table(data[['Price']],index=['Date_of_Journey'],aggfunc='sum').plot(kind='bar',figsize=(10,5))
plt.xlabel('Date_of_Journey')
plt.ylabel('Price')


# In[25]:


data.pivot_table(data[['Price']],index=['Source','Destination'],aggfunc='sum').plot(kind='bar',figsize=(10,5))
plt.xlabel('Source')
plt.ylabel('Price')


# In[26]:


data.pivot_table(data[['Price']],index=['Total_Stops'],aggfunc='sum').plot(kind='bar',figsize=(10,5))
plt.xlabel('Duration')
plt.ylabel('Price')


# In[27]:


data.pivot_table(data[['Price']],index=['Additional_Info'],aggfunc='sum').plot(kind='bar',figsize=(10,5))
plt.xlabel('Source')
plt.ylabel('Price')


# ## FEATURE ENGINEERING

# In[28]:


data['Date_of_Journey']=pd.to_datetime(data['Date_of_Journey'])


# In[29]:


# changing No_Info column in Additional_Info to 'no_info'

data[data['Additional_Info']=='No Info']


# In[30]:


data['Additional_Info'].replace('No Info', 'No info',inplace=True)


# In[31]:


# creating day and month columns from the Date of Journey field.

import datetime as dt

data['DOJ_Month']=data['Date_of_Journey'].dt.month
data['DOJ_Day']=data['Date_of_Journey'].dt.day
data=data.drop(['Date_of_Journey'],axis=1)


# In[32]:


data.head()


# In[33]:


# treating departure time and arrival time

data['Dep_Hour']=pd.to_datetime(data['Dep_Time']).dt.hour
data['Dep_Minute']=pd.to_datetime(data['Dep_Time']).dt.minute
data=data.drop(['Dep_Time'],axis=1)

data['Arrival_Hour']=pd.to_datetime(data['Arrival_Time']).dt.hour
data['Arrival_Minute']=pd.to_datetime(data['Arrival_Time']).dt.minute
data=data.drop(['Arrival_Time'],axis=1)


# In[34]:


data.head()


# In[35]:


# correcting the 5m data entry of Duration column

data.loc[data['Duration']== '5m', 'Duration'] = '1h 25m'


# In[36]:


data['Duration_hours'] =  data['Duration'].apply(lambda x: str(x).split('h')[0]).astype('int')
data['Duration_minutes'] = data['Duration'].apply(lambda x: 0 if ((str(x).split('h')[1].split('m')[0])=='') else str(x).split('h')[1].split('m')[0]).astype('int')


# In[37]:


data=data.drop('Duration',axis=1)


# In[38]:


data.Duration_hours.unique()


# In[39]:


data.Duration_minutes.unique()


# In[40]:


data.info()


# In[41]:


# treating the Total_Stops column
data['Total_Stops'].str.split(' ')
data['Stops'] = data['Total_Stops'].apply(lambda x: 0 if (str(x).split(' ')[0] =='non-stop') else (str(x).split(' ')[0])).astype('int')


# In[42]:


data=data.drop('Total_Stops',axis=1)


# In[43]:


data.head(5)


# In[44]:


data.info()


# In[45]:


data.shape


# ## FEATURE SELECTION

# In[46]:


X=data.drop(['Price'],axis=1)
y=data['Price']
print(X.shape,y.shape)


# In[47]:


# separating the date into numerical, nominal and categorical fields

cat=['Airline','Source','Destination','Route','Additional_Info']
num=['DOJ_Month','DOJ_Day','Dep_Hour','Dep_Minute','Arrival_Hour','Arrival_Minute','Duration_hours','Duration_minutes','Stops']


# In[48]:


X_num,X_cat=X[num],X[cat]


# In[49]:


# Applying Pearson's correlation on X_num

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# feature selection
fs = SelectKBest(score_func=f_regression, k=2)
# learn relationship from training data
fs.fit(X_num, y)
# transform train input data
X_num_fs = fs.transform(X_num)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()


# In[50]:


import pickle 
pickle.dump(fs,open('Pearson_corr','wb'))


# In[50]:


# Applying ANOVA F-Test on X_cat

from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

oe=OrdinalEncoder()
X_cat_oe=oe.fit_transform(X_cat).astype('int')

fs = SelectKBest(score_func=f_classif, k='all')
# learn relationship from training data
fs.fit(X_cat_oe, y)
# transform train input data
X_cat_fs = fs.transform(X_cat)
# transform
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()


# In[52]:


pickle.dump(fs,open('ANOVA_cat','wb'))


# In[51]:


X_new=pd.concat([pd.DataFrame(X_num_fs),pd.DataFrame(X_cat_fs)],axis=1)
print(X_new.shape)


# In[52]:


X_new.info()


# ### ONE HOT ENCODING

# In[53]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False, drop='first')
X_new_ohe = ohe.fit_transform(X_cat_fs)
print(X_new_ohe.shape)


# In[56]:


pickle.dump(ohe,open('OneHotEncoder','wb'))


# In[54]:


X_final = np.concatenate([X_num_fs, X_new_ohe],axis=1)
print(X_final.shape)


# ### TRAIN TEST SPLIT

# In[55]:


from sklearn.model_selection import train_test_split

X_temp,X_test,y_temp,y_test=train_test_split(X_final,y,test_size=0.1) 
X_train,X_val,y_train,y_val=train_test_split(X_temp,y_temp,test_size=0.2)
print(X_train.shape,X_val.shape,X_test.shape,y_train.shape,y_val.shape,y_test.shape)


# ## DATA SCALING

# ### Scaling of training, testing and validation dataset features

# In[56]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler(feature_range=(0,1))
mms.fit(X_train)
X_train_scaled = mms.transform(X_train)
X_val_scaled = mms.transform(X_val)
X_test_scaled = mms.transform(X_test)
print(X_train_scaled.shape,X_val.shape,X_test_scaled.shape)


# In[60]:


pickle.dump(mms,open('MinMaxScalar','wb'))


# ### Scaling of target variable

# In[57]:


# Min max scaler

from sklearn.preprocessing import MinMaxScaler

mms1=MinMaxScaler(feature_range=(0,1))
mms1.fit(y_train.to_numpy().reshape(-1,1))
y_train_scaled = mms1.transform(y_train.to_numpy().reshape(-1,1)).flatten()
y_val_scaled = mms1.transform(y_val.to_numpy().reshape(-1,1)).flatten()
y_test_scaled = mms1.transform(y_test.to_numpy().reshape(-1,1)).flatten()
print(y_train_scaled.shape,y_val_scaled.shape,y_test_scaled.shape)


# ### CHECK FOR DATA SKEWNESS

# In[58]:


from scipy.stats import skew
y_train_skew = skew(y_train,axis=0)
print(y_train_skew)


# In[59]:


y_train_log_skew = skew(np.log(y_train+1),axis=0)
print(y_train_log_skew)


# In[60]:


y.value_counts()


# In[61]:


y_train.value_counts()


# ## MODELLING

# ### I. REGRESSION BASED ALGORITHMS

# ### 1. Linear Regression

# In[66]:


X_train_lr=np.concatenate([X_train_scaled,X_val_scaled],axis=0)
print(X_train_lr.shape)


# In[67]:


y_train_lr = np.concatenate([y_train,y_val],axis=0)
print(y_train_lr.shape)


# In[68]:


from sklearn.linear_model import LinearRegression

linreg=LinearRegression()
linreg.fit(X_train_lr,y_train_lr)


# In[69]:


y_pred_lr=linreg.predict(X_test_scaled)
print(y_pred_lr.shape)


# In[70]:


pd.Series(y_pred_lr).value_counts()


# In[71]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
test_mae_lr = mean_absolute_error(y_test,y_pred_lr)
test_mse_lr = mean_squared_error(y_test,y_pred_lr)
test_r2_square_lr= r2_score(y_test,y_pred_lr)
print(test_mse_lr)
print(test_mae_lr)
print(test_r2_square_lr)


# ### 2. Support Vector Regressor

# In[72]:


X_train_svr=np.concatenate([X_train_scaled,X_val_scaled],axis=0)
print(X_train_svr.shape)


# In[73]:


y_train_svr = np.concatenate([y_train,y_val],axis=0)
print(y_train_svr.shape)


# In[74]:


from sklearn.svm import SVR
svr=SVR()
svr.fit(X_train_svr,y_train_svr)


# In[75]:


y_pred_svr=svr.predict(X_test_scaled)
print(y_pred_svr.shape)


# In[76]:


pd.Series(y_pred_svr).value_counts()


# In[77]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
test_mae_svr = mean_absolute_error(y_test,y_pred_svr)
test_mse_svr = mean_squared_error(y_test,y_pred_svr)
test_r2_square_svr= r2_score(y_test,y_pred_svr)
print(test_mse_svr)
print(test_mae_svr)
print(test_r2_square_svr)


# ### 3. K- Neighbour Regressor

# In[78]:


X_train_knr=np.concatenate([X_train_scaled,X_val_scaled],axis=0)
print(X_train_knr.shape)


# In[79]:


y_train_knr = np.concatenate([y_train,y_val],axis=0)
print(y_train_knr.shape)


# In[80]:


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(X_train_knr,y_train_knr)


# In[81]:


y_pred_knr=knr.predict(X_test_scaled)
print(y_pred_knr.shape)


# In[82]:


pd.Series(y_pred_knr).value_counts()


# In[83]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
test_mae_knr = mean_absolute_error(y_test,y_pred_knr)
test_mse_knr = mean_squared_error(y_test,y_pred_knr)
test_r2_square_knr= r2_score(y_test,y_pred_knr)
print(test_mse_knr)
print(test_mae_knr)
print(test_r2_square_knr)


# ### 4. Decision Tree Regressor

# In[84]:


X_train_dtr=np.concatenate([X_train_scaled,X_val_scaled],axis=0)
print(X_train_dtr.shape)


# In[85]:


y_train_dtr = np.concatenate([y_train,y_val],axis=0)
print(y_train_dtr.shape)


# In[86]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train_dtr,y_train_dtr)


# In[87]:


y_pred_dtr=dtr.predict(X_test_scaled)
print(y_pred_dtr.shape)


# In[88]:


pd.Series(y_pred_dtr).value_counts()


# In[89]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
test_mae_dtr = mean_absolute_error(y_test,y_pred_dtr)
test_mse_dtr = mean_squared_error(y_test,y_pred_dtr)
test_r2_square_dtr= r2_score(y_test,y_pred_dtr)
print(test_mse_dtr)
print(test_mae_dtr)
print(test_r2_square_dtr)


# ### 5. Bagging Regressor

# In[90]:


X_train_br=np.concatenate([X_train_scaled,X_val_scaled],axis=0)
print(X_train_dtr.shape)


# In[91]:


y_train_br = np.concatenate([y_train,y_val],axis=0)
print(y_train_dtr.shape)


# In[92]:


from sklearn.ensemble import BaggingRegressor
br = BaggingRegressor()
br.fit(X_train_br,y_train_br)


# In[93]:


y_pred_br=br.predict(X_test_scaled)
print(y_pred_br.shape)


# In[94]:


pd.Series(y_pred_br).value_counts()


# In[95]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
test_mae_br = mean_absolute_error(y_test,y_pred_br)
test_mse_br = mean_squared_error(y_test,y_pred_br)
test_r2_square_br= r2_score(y_test,y_pred_br)
print(test_mse_br)
print(test_mae_br)
print(test_r2_square_br)


# ### II. DECISION TREES BASED ALGORITHMS

# ### 1. Random Forest Regressor

# In[119]:


## AUTO ML

from flaml import AutoML
auto_model = AutoML()
auto_model.fit(X_train = X_train_scaled, y_train = y_train, X_val = X_val_scaled, y_val= y_val, 
          task = 'regression' , # 'regression','classification' 
          metric = 'mse' , 
          eval_method = 'holdout', # 'cv' , 'holdout' 
          # n_split = 5, # used with 'cv'
          # split_type = 'uniform' # 'stratified' , 'uniform' used with 'cv'
          estimator_list = ['rf'] , # regression - 'rf', 'lgbm', 'xgboost', 'catboost', 'xgb_limit_depth', 'extra_tree'
                                  # classification - 'lrl1','lrl2' (additional)    
          # ensemble = True, # model emsembling
          # sample_weight = 'balanced' # used with imbalanced targets
          time_budget = 1200 # in seconds 
         )

model= auto_model.model # selects best model      


# In[120]:


import pickle 
pickle.dump(model,open('model_rf_ML','wb'))


# In[96]:


import pickle
model_rf=pickle.load(open('model_rf_ML','rb'))


# In[97]:


y_pred_rf=model_rf.predict(X_test_scaled)
print(y_pred_rf.shape)


# In[98]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
test_mae_rf_ML = mean_absolute_error(y_test,y_pred_rf)
test_mse_rf_ML = mean_squared_error(y_test,y_pred_rf)
test_r2_square_ML=r2_score(y_test,y_pred_rf)
print(test_mse_rf_ML)
print(test_mae_rf_ML)
print(test_r2_square_ML)


# ### 2. XGBoost Regressor

# In[124]:


## AUTO ML

from flaml import AutoML
auto_model = AutoML()
auto_model.fit(X_train = X_train_scaled, y_train = y_train, X_val = X_val_scaled, y_val= y_val, 
          task = 'regression' , # 'regression','classification' 
          metric = 'mse' , 
          eval_method = 'holdout', # 'cv' , 'holdout' 
          # n_split = 5, # used with 'cv'
          # split_type = 'uniform' # 'stratified' , 'uniform' used with 'cv'
          estimator_list = ['xgboost'] , # regression - 'rf', 'lgbm', 'xgboost', 'catboost', 'xgb_limit_depth', 'extra_tree'
                                  # classification - 'lrl1','lrl2' (additional)    
          # ensemble = True, # model emsembling
          # sample_weight = 'balanced' # used with imbalanced targets
          time_budget = 1200 # in seconds 
         )

model1= auto_model.model # selects best model      


# In[125]:


import pickle 
pickle.dump(model1,open('model_xgboost_ML','wb'))


# In[62]:


import pickle
model_xg_ML=pickle.load(open('model_xgboost_ML','rb'))


# In[63]:


y_pred_xg_ML=model_xg_ML.predict(X_test_scaled)
print(y_pred_xg_ML.shape)


# In[64]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
test_mae_xgboost_ML = mean_absolute_error(y_test,y_pred_xg_ML)
test_mse_xgboost_ML = mean_squared_error(y_test,y_pred_xg_ML)
test_r2_score_xgboost_ML = r2_score(y_test,y_pred_xg_ML)
print(test_mse_xgboost_ML)
print(test_mae_xgboost_ML)
print(test_r2_score_xgboost_ML)


# ### 3. Light Gradient Boosted Machine Regressor

# In[129]:


## AUTO ML

from flaml import AutoML
auto_model = AutoML()
auto_model.fit(X_train = X_train_scaled, y_train = y_train, X_val = X_val_scaled, y_val= y_val, 
          task = 'regression' , # 'regression','classification' 
          metric = 'mse' , 
          eval_method = 'holdout', # 'cv' , 'holdout' 
          # n_split = 5, # used with 'cv'
          # split_type = 'uniform' # 'stratified' , 'uniform' used with 'cv'
          estimator_list = ['lgbm'] , # regression - 'rf', 'lgbm', 'xgboost', 'catboost', 'xgb_limit_depth', 'extra_tree'
                                  # classification - 'lrl1','lrl2' (additional)    
          # ensemble = True, # model emsembling
          # sample_weight = 'balanced' # used with imbalanced targets
          time_budget = 1200 # in seconds 
         )

model2= auto_model.model # selects best model      


# In[130]:


import pickle 
pickle.dump(model2,open('model_lgbm_ML','wb'))


# In[65]:


import pickle
model_lgbm_AutoML=pickle.load(open('model_lgbm_ML','rb'))


# In[66]:


y_pred_lgbm_ML=model_lgbm_AutoML.predict(X_test_scaled)
print(y_pred_lgbm_ML.shape)


# In[67]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
test_mae_lgbm_ML = mean_absolute_error(y_test,y_pred_lgbm_ML)
test_mse_lgbm_ML = mean_squared_error(y_test,y_pred_lgbm_ML)
test_r2_score_ML = r2_score(y_test,y_pred_lgbm_ML)
print(test_mse_lgbm_ML)
print(test_mae_lgbm_ML)
print(test_r2_score_ML)


# ### 4. Extra Tree Regressor

# In[135]:


## AUTO ML

from flaml import AutoML
auto_model = AutoML()
auto_model.fit(X_train = X_train_scaled, y_train = y_train, X_val = X_val_scaled, y_val= y_val, 
          task = 'regression' , # 'regression','classification' 
          metric = 'mse' , 
          eval_method = 'holdout', # 'cv' , 'holdout' 
          # n_split = 5, # used with 'cv'
          # split_type = 'uniform' # 'stratified' , 'uniform' used with 'cv'
          estimator_list = ['extra_tree'] , # regression - 'rf', 'lgbm', 'xgboost', 'catboost', 'xgb_limit_depth', 'extra_tree'
                                  # classification - 'lrl1','lrl2' (additional)    
          # ensemble = True, # model emsembling
          # sample_weight = 'balanced' # used with imbalanced targets
          time_budget = 1200 # in seconds 
         )

model3= auto_model.model # selects best model      


# In[136]:


import pickle 
pickle.dump(model3,open('model_extra_tree_ML','wb'))


# In[105]:


import pickle
model_extra_tree_AutoML=pickle.load(open('model_extra_tree_ML','rb'))


# In[106]:


y_pred_extra_tree_ML=model_extra_tree_AutoML.predict(X_test_scaled)
print(y_pred_extra_tree_ML.shape)


# In[107]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
test_mae_extra_tree_ML = mean_absolute_error(y_test,y_pred_extra_tree_ML)
test_mse_extra_tree_ML = mean_squared_error(y_test,y_pred_extra_tree_ML)
test_r2_score_extra_tree_ML=r2_score(y_test,y_pred_extra_tree_ML)
print(test_mse_extra_tree_ML)
print(test_mae_extra_tree_ML)
print(test_r2_score_extra_tree_ML)


# ### III. NEURAL NETWORKS (ANN)

# In[140]:


import autokeras as ak
import tensorflow as tf
import keras_tuner

tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)

input1 = ak.StructuredDataInput()
output = ak.DenseBlock()(input1)
output = ak.RegressionHead()(output)
model = ak.AutoModel(inputs=input1, outputs=output, max_trials=100, overwrite=True,metrics=['RootMeanSquaredError'])


# In[141]:


# Neural Network Training

model.fit(X_train, np.log10(y_train+1), validation_data = (X_val, np.log10(y_val+1))) # RMSLE
model = model.export_model()
model.summary()


# In[142]:


model.save('model_ak')


# In[68]:


import autokeras as ak
from tensorflow.keras.models import load_model
model_ak=load_model('model_ak',custom_objects=ak.CUSTOM_OBJECTS)


# In[69]:


y_pred_ak = model_ak.predict(X_test)


# In[70]:


y_pred_ak.shape


# In[71]:


import math
for i in range (0,len(y_pred_ak)):
    y_pred_ak[i] = pow(10,y_pred_ak[i])-1 


# In[72]:


pd.DataFrame(y_pred_ak).value_counts()


# In[73]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
test_mae_ak = mean_absolute_error(y_test,y_pred_ak)
test_mse_ak = mean_squared_error(y_test,y_pred_ak)
test_r2_score_ak=r2_score(y_test,y_pred_ak)
print(test_mse_ak)
print(test_mae_ak)
print(test_r2_score_ak)


# ### Ensemble of best three models

# In[74]:


y_pred_ensemble = (y_pred_ak + y_pred_lgbm_ML.reshape(-1,1) + y_pred_xg_ML.reshape(-1,1))/3


# In[75]:


y_pred_ensemble.shape


# In[76]:


y_pred_ensemble =  y_pred_ensemble.flatten()


# In[77]:


pd.Series(y_pred_ensemble).value_counts()


# In[78]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
test_mae_ensemble= mean_absolute_error(y_test,y_pred_ensemble)
test_mse_ensemble = mean_squared_error(y_test,y_pred_ensemble)
test_r2_square_ensemble = r2_score(y_test,y_pred_ensemble)

print(test_mse_ensemble)
print(test_mae_ensemble)
print(test_r2_square_ensemble)


# ## APP: FLASK

# In[80]:


# Load libraries
import flask
from flask import Flask, request, render_template
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import autokeras as ak
import pandas as pd
import numpy as np
import pickle

#initialize app
app = Flask(__name__)

#Load NN models
model_ak = load_model('model_ak',custom_objects=ak.CUSTOM_OBJECTS)


# Load ML models
xg = pickle.load(open('model_xgboost_ML', 'rb'))
lgbm = pickle.load(open('model_lgbm_ML', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    airline = str(request.form.get("airline"))
    doj = str(request.form.get("date_of_journey"))
    source = str(request.form.get("source"))
    dest = str(request.form.get("destination"))
    route = str(request.form.get("route"))
    dep_time = str(request.form.get("dep_time"))
    arrival_time = str(request.form.get("arrival_time"))
    duration = str(request.form.get("duration"))
    total_stops = str(request.form.get("total_stops"))
    additional_info = str(request.form.get("additional_info"))
    
    x_new = pd.DataFrame(np.zeros((1,10)),columns = ['airline','doj','source','dest','route','dep_time','arrival_time','duration','total_stops','additional_info'])
    x_new.iloc[0,0] = airline
    x_new.iloc[0,1] = doj
    x_new.iloc[0,2] = source
    x_new.iloc[0,3] = dest
    x_new.iloc[0,4] = route
    x_new.iloc[0,5] = dep_time
    x_new.iloc[0,6] = arrival_time
    x_new.iloc[0,7] = duration
    x_new.iloc[0,8] = total_stops
    x_new.iloc[0,9] = additional_info
    
    # FEATURE ENGINEERING
    x_new['doj']=pd.to_datetime(x_new['doj'])
    #x_new['additional_info'].replace('No Info', 'No info',inplace=True)
    
    x_new['DOJ_Month']=x_new['doj'].dt.month
    x_new['DOJ_Day']=x_new['doj'].dt.day
    x_new=x_new.drop(['doj'],axis=1)
    
    x_new['Dep_Hour']=pd.to_datetime(x_new['dep_time']).dt.hour
    x_new['Dep_Minute']=pd.to_datetime(x_new['dep_time']).dt.minute
    x_new=x_new.drop(['dep_time'],axis=1)

    x_new['Arrival_Hour']=pd.to_datetime(x_new['arrival_time']).dt.hour
    x_new['Arrival_Minute']=pd.to_datetime(x_new['arrival_time']).dt.minute
    x_new=x_new.drop(['arrival_time'],axis=1)
    
    #x_new.loc[x_new['duration']== '5m', 'duration'] = '1h 25m'
    x_new['Duration_hours'] =  x_new['duration'].apply(lambda x: str(x).split('h')[0]).astype('int')
    x_new['Duration_minutes'] = x_new['duration'].apply(lambda x: 0 if ((str(x).split('h')[1].split('m')[0])=='') else str(x).split('h')[1].split('m')[0]).astype('int')
    x_new=x_new.drop('duration',axis=1)
    
    x_new['total_stops'].str.split(' ')
    x_new['Stops'] = x_new['total_stops'].apply(lambda x: 0 if (str(x).split(' ')[0] =='non-stop') else (str(x).split(' ')[0])).astype('int')
    x_new=x_new.drop('total_stops',axis=1)
    
     ## FEATURE SELECTION
    
    cat=['airline','source','dest','route','additional_info']
    num=['DOJ_Month','DOJ_Day','Dep_Hour','Dep_Minute','Arrival_Hour','Arrival_Minute','Duration_hours','Duration_minutes','Stops']
    
    x_new_num,x_new_cat = x_new[num],x_new[cat]
    
    fs_num = pickle.load(open('Pearson_corr','rb'))
    x_new_num_fs = fs_num.transform(x_new_num)
    
    fs_cat = pickle.load(open('ANOVA_cat','rb'))
    x_new_cat_fs = fs_cat.transform(x_new_cat)
    
    x_new_concat = pd.concat([pd.DataFrame(x_new_num_fs),pd.DataFrame(x_new_cat_fs)],axis=1)
    print(x_new_concat.shape)
    
    ## ONE HOT ENCODING

    ohe_new = pickle.load(open('OneHotEncoder','rb'))
    x_new_ohe = ohe_new.transform(x_new_cat_fs)
    x_new_final= np.concatenate([x_new_num_fs, x_new_ohe],axis=1)
    print(x_new_final.shape)
    
    ## DATA SCALING
    mms_new = pickle.load(open('MinMaxScalar','rb'))
    x_new_mms = mms_new.transform(x_new_final)
    
    ## PREDICTION  
    
    model_ak_new = load_model('model_ak',custom_objects=ak.CUSTOM_OBJECTS)
    y_pred_ak_new = model_ak_new.predict(x_new_final)
    y_pred_ak_x_new = pow(10,y_pred_ak_new)-1 
    
    model_xg = pickle.load(open('model_xgboost_ML', 'rb'))
    y_pred_xg_x_new = model_xg.predict(x_new_mms)
     
    model_lgbm = pickle.load(open('model_lgbm_ML', 'rb'))
    y_pred_lgbm_x_new = model_lgbm.predict(x_new_mms)
    
    y_pred_ensemble_new = (y_pred_ak_x_new + y_pred_xg_x_new.reshape(-1,1) +  y_pred_lgbm_x_new.reshape(-1,1))/3
    y_pred_ensemble_new =  y_pred_ensemble_new.flatten()
    
    print("Predicted Price: ",y_pred_ensemble_new)
    
    # Output
    output = int(np.round(y_pred_ensemble_new[0]))
    return render_template('index.html',prediction_text ='Predicted Ticket Price: Rs {}'.format(output))
# run app    
if __name__ == "__main__":
    app.run()

