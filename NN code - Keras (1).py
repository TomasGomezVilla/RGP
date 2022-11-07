#!/usr/bin/env python
# coding: utf-8

# In[4]:


### Neural networks with multiple entries - Reference "Le Machine Learning avec Python: De la théorie à la pratique" par Patrick Albert
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
#command to avoid tf print on standard errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# In[73]:


df = pd.read_excel(r'C:\Users\tgome\Documents\Tomas Gomez Villa\2022-1 RSB\RGP\Raw_Data_Last_Version.xlsx', sheet_name='ETH_Twitter_All',header=1)


# In[74]:


df.shape


# In[75]:


na_count=df.isna()
na_count.sum()


# In[76]:


df=df.dropna()
na_count=df.isna()
na_count.sum()


# In[77]:


x_data=df[['Positive', 'Neutral', 'Negative','Energy consumption', 'Transaction fees']]
y_data = df[['Volume']]

x_plot=df[['DateTime']]
x_data.shape


# In[79]:


plt.plot(x_plot,y_data)
plt.title('ETH volume')
plt.show


# In[12]:


def Split_Train_Test(data, test_ratio):
    '''splits data into a training and testing set'''
    train_set_size = 1 - int(len(data) * test_ratio)
    train_set = data[:train_set_size]
    test_set = data[train_set_size:]
    return train_set, test_set


# In[80]:


#Train test split - local function
#x_train, x_test = Split_Train_Test(x_data, 0.3)
#y_train, y_test = Split_Train_Test(y_data, 0.3)

#Train test split - SKlearn
all_x_train, x_test, all_y_train, y_test = train_test_split(x_data, y_data, shuffle=False)
x_train, x_validation, y_train, y_validation = train_test_split(all_x_train, all_y_train,test_size=0.3,shuffle=False)


# In[34]:


y_train


# In[81]:


#scaling must be performed given the different nature of used variables
scaler=StandardScaler()


# In[82]:


x_train_scaled = scaler.fit_transform(x_train)
x_validation_scaled = scaler.transform(x_validation)
x_test_scaled=scaler.transform(x_test)


# In[83]:


x_train_scaled.shape


# In[84]:


x_validation_scaled[:,3]


# In[85]:


#This code allocates the SA data to the first input neurons, then the transaction en electricity consumption variables will be handled by different neurons. 
input_1= keras.layers.Input(shape=x_train_scaled.shape[1:])
input_2= keras.layers.Input(shape=[3])
input_3= keras.layers.Input(shape=[1])
input_4=keras.layers.Input(shape=[1])
#A separation has been made between taking the entire data as a whole - input_0, and taking clusters of data. This will facilitate evaluating the effectiveness of our NN.


# In[86]:



hidden1=keras.layers.Dense(80,activation='relu')(input_1)
hidden2=keras.layers.Dense(40,activation='relu')(hidden1)
concat_1=keras.layers.concatenate([input_2,hidden2])
hidden3=keras.layers.Dense(30,activation='relu')(concat_1)

concat_2=keras.layers.concatenate([input_3,hidden3])
hidden4=keras.layers.Dense(30,activation='relu')(concat_2)
concat_3=keras.layers.concatenate([input_4,hidden4])

#This code allows to process data in an independent manner - one hidden layer per input.
#Also, by reintroducing the entirety of data, overfitting of data clusters will be mitigated to some extent
#Then all results are compiled to access to its results. 
output_1=keras.layers.Dense(1)(concat_3)
#results will only have one neurone to run a linear regression now that we are comparing results.


# In[87]:


model=keras.models.Model(inputs=[input_1,input_2,input_3, input_4],outputs=[output_1])


# In[88]:


model.compile(loss='mse',optimizer='rmsprop',metrics=['mae'])


# In[89]:


#preparing data inputs for training
x_train_2=x_train_scaled[:,:3]
x_train_3=x_train_scaled[:,3]
x_train_4=x_train_scaled[:,4]

#preparing data inputs for validation
x_validation_2=x_validation_scaled[:,:3]
x_validation_3=x_validation_scaled[:,3]
x_validation_4=x_validation_scaled[:,4]

#preparing data inputs for testing
x_test_2=x_test_scaled[:,:3]
x_test_3=x_test_scaled[:,3]
x_test_4=x_test_scaled[:,4]


# In[90]:


train_results = model.fit((x_train_scaled, x_train_2, x_train_3,x_train_4), [y_train,y_train,y_train],epochs=5000,validation_data=((x_validation_scaled,x_validation_2,x_validation_3,x_validation_4),[y_validation,y_validation,y_validation]))


# In[91]:


res_eval=model.evaluate((x_test_scaled,x_test_2,x_test_3,x_test_4),[y_test])


# In[92]:


x_new_1,x_new_2,x_new_3,x_new_4 = x_test_scaled[:30],x_test_scaled[:30,:3], x_test_scaled[:30,3],x_test_scaled[:30,4]
#the first three lines are taken to perform a prediction


# In[93]:


x_new_4.shape


# In[94]:


y_pred = model.predict((x_new_1,x_new_2,x_new_3,x_new_4))
print(f"y_pred= {y_pred}")


# In[95]:


weights = model.get_weights()
weights


# In[65]:


for lay in model.layers:
    print(lay.name)
    print(lay.get_weights())


# In[96]:


model.summary()

