#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[2]:


train_data=pd.read_excel(r"apstrain.xlsx")


# In[3]:


train_data


# In[4]:


pd.set_option('display.max_columns',None)


# In[5]:


train_data.shape


# In[6]:


train_data.info()


# In[7]:


train_data['Duration'].value_counts()


# In[8]:


train_data.dropna(inplace=True)


# In[9]:


train_data.isnull().sum()


# # EXPLARATORY DATA ANALYSIS

# In[10]:


train_data["Journey_day"]=pd.to_datetime(train_data.Date_of_Journey,format="%d/%m/%Y").dt.day


# In[11]:


train_data["Journey_month"]=pd.to_datetime(train_data["Date_of_Journey"],format="%d/%m/%Y").dt.month


# In[12]:


train_data


# In[13]:


train_data.drop(["Date_of_Journey"],axis=1,inplace=True)


# In[14]:


train_data


# In[15]:


train_data["Departure_hour"]=pd.to_datetime(train_data.Dep_Time).dt.hour
train_data["Departure_min"]=pd.to_datetime(train_data.Dep_Time).dt.minute
train_data.drop(["Dep_Time"],axis=1,inplace=True)


# In[16]:


train_data.head(5)


# In[17]:


train_data["Arrival_hour"]=pd.to_datetime(train_data.Arrival_Time).dt.hour
train_data["Arrival_min"]=pd.to_datetime(train_data.Arrival_Time).dt.minute
train_data.drop(["Arrival_Time"],axis=1,inplace=True)


# In[18]:


train_data.head(3)


# In[19]:


duration = list(train_data["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:   
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   
        else:
            duration[i] = "0h " + duration[i]         


# In[20]:


duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0])) 
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))


# In[21]:


train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins


# In[22]:


train_data


# In[23]:


train_data.drop("Duration",axis=1,inplace=True)


# In[24]:


train_data


# # HANDLING CATEGORCAL DATA

# In[25]:


feature=[feature for feature in train_data.columns if(train_data.columns).dtype=='O']


# In[26]:


feature


# In[27]:


train_data['Airline'].value_counts()


# In[28]:


sns.catplot(x='Airline',y='Price',data=train_data.sort_values("Price",ascending=False),kind="boxen", height = 6, aspect = 3)
plt.show()


# In[29]:


Airline=train_data.Airline
Airline=pd.get_dummies(Airline,drop_first=True)


# In[30]:


Airline.head(3)


# In[31]:


train_data['Source'].value_counts()


# In[32]:


sns.catplot(y='Price',x='Source',data=train_data.sort_values('Price',ascending=False),kind='boxen',height=6,aspect=3)
plt.show()


# In[33]:


Source=train_data[['Source']]
Source=pd.get_dummies(Source,drop_first=True)


# In[34]:


Source.head()


# In[35]:


train_data['Destination'].value_counts()


# In[36]:


sns.catplot(x="Destination",y="Price",data=train_data.sort_values("Price",ascending=True),kind="boxen",height=6,aspect=3)


# In[37]:


Destination=train_data.Destination
Destination=pd.get_dummies(Destination,drop_first=True)
Destination


# In[38]:


train_data["Route"]


# In[39]:


train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[40]:


train_data["Total_Stops"].value_counts()


# In[41]:


train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[42]:


train_data.head()


# In[43]:


data_train = pd.concat([train_data, Airline, Source, Destination], axis = 1)


# In[44]:


data_train.head()


# In[45]:


data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[46]:


data_train.head()


# In[47]:


data_train.shape


# # TEST DATA 

# In[48]:


test_data=pd.read_excel('apstest.xlsx')


# In[49]:


test_data


# # PREPROCESSING

# In[50]:


test_data.isna().sum()


# In[51]:


test_data.info()


# # EDA

# In[52]:


test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[53]:


test_data


# In[54]:


test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)


# In[55]:


test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[56]:


test_data


# In[57]:


duration = list(test_data["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:   
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   
        else:
            duration[i] = "0h " + duration[i]


# In[58]:


duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))


# In[59]:


test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# # CATEGORICAL DATA

# In[60]:


print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)


# In[61]:


print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)


# In[62]:


print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)


# In[63]:


test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[64]:


test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[65]:


test_data = pd.concat([test_data, Airline, Source, Destination], axis = 1)

test_data.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[66]:


test_data.shape


# In[67]:


test_data


# # FEATURE SELECTION

# In[68]:


data_train.shape


# In[69]:


data_train.columns


# In[73]:


X=data_train.loc[:,['Total_Stops','Journey_day', 'Journey_month',
       'Departure_hour', 'Departure_min', 'Arrival_hour', 'Arrival_min',
       'Duration_hours', 'Duration_mins', 'Air India', 'GoAir', 'IndiGo',
       'Jet Airways', 'Jet Airways Business', 'Multiple carriers',
       'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara',
       'Vistara Premium economy', 'Source_Chennai', 'Source_Delhi',
       'Source_Kolkata', 'Source_Mumbai', 'Cochin', 'Delhi', 'Hyderabad',
       'Kolkata', 'New Delhi']]


# In[75]:


X.head()


# In[76]:


y = data_train.iloc[:, 1]
y.head()


# In[81]:


plt.figure(figsize=(20,20))
sns.heatmap(train_data.corr(),annot=True,cmap="RdYlGn")
plt.show()


# In[84]:


from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)


# In[86]:


model.feature_importances_


# In[93]:


plt.figure(figsize = (12,8))
imp_features = pd.Series(model.feature_importances_, index=X.columns)
imp_features.nlargest(20).plot(kind='bar')
plt.show()


# # FITTING MODEL USING RANDOM FOREST

# In[94]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[95]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)


# In[96]:


y_pred = rf.predict(X_test)


# In[97]:


rf.score(X_train, y_train)


# In[98]:


rf.score(X_test, y_test)


# In[99]:


sns.distplot(y_test-y_pred)
plt.show()


# In[100]:


plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[101]:


from sklearn import metrics


# In[102]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[103]:


# RMSE/(max(DV)-min(DV))

2090.5509/(max(y)-min(y))


# In[104]:


metrics.r2_score(y_test, y_pred)


# # Hyperparameter Tuning

# In[105]:


from sklearn.model_selection import RandomizedSearchCV


# In[106]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[107]:


# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[109]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator =rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[110]:


rf_random.fit(X_train,y_train)


# In[111]:


rf_random.best_params_


# In[112]:


prediction = rf_random.predict(X_test)


# In[113]:


plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()


# In[114]:


plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[115]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# # SAVING THE MODEL

# In[122]:


import pickle
# open a file, where you ant to store the data
file = open('flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[123]:


model = open('flight_rf.pkl','rb')
forest = pickle.load(model)


# In[124]:


y_prediction = forest.predict(X_test)


# In[125]:


metrics.r2_score(y_test, y_prediction)


# In[ ]:




