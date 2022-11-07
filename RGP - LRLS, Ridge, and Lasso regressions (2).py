#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error


# In[102]:


df = pd.read_excel(r'C:\Users\tgome\Documents\Tomas Gomez Villa\2022-1 RSB\RGP\Raw_Data_Last_Version.xlsx', sheet_name='ETH_Twitter',header=1)


# In[ ]:





# In[103]:


na_count=df.isna()
na_count.sum()


# In[104]:


df=df.dropna()
na_count=df.isna()
na_count.sum()


# In[5]:


df.columns


# In[74]:


df.describe


# In[105]:


plt.figure(figsize = (10, 10))
sns.heatmap(df.corr(), annot = True)


# In[106]:


df.columns
x=df[['Positive', 'Neutral', 'Negative']]

positive=df[['Positive']]
negative=df[['Negative']]
x.shape


# In[107]:


y=df[['Volume']]
y


# In[68]:


features = df.columns[2:5]
features.shape


# In[25]:


# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(df.DateTime,
        df.Volume,
        color="black")
# set x-axis label
ax.set_xlabel("time", fontsize = 14)
# set y-axis label
ax.set_ylabel("BTC volume",
              color="black",
              fontsize=14)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(df.DateTime, df.Neutral,color="blue")
ax2.set_ylabel("Neutral",color="blue",fontsize=14)
plt.title('BTC and Twitter neutral mentions')
plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')


# In[26]:


# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(df.DateTime,
        df.Volume,
        color="black")
# set x-axis label
ax.set_xlabel("time", fontsize = 14)
# set y-axis label
ax.set_ylabel("BTC volume",
              color="black",
              fontsize=14)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(df.DateTime, df.Negative,color="red")
ax2.set_ylabel("Negative",color="red",fontsize=14)
plt.title('BTC and Twitter negative mentions')
plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')


# In[13]:


def Split_Train_Test(data, test_ratio):
    '''splits data into a training and testing set'''
    train_set_size = 1 - int(len(data) * test_ratio)
    train_set = data[:train_set_size]
    test_set = data[train_set_size:]
    return train_set, test_set


# In[108]:


x_train, x_test = Split_Train_Test(x, 0.3)


# In[109]:


y_train, y_test = Split_Train_Test(y, 0.3)


# In[110]:


scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[111]:


lr = LinearRegression()
lr.fit(x_train, y_train)
prediction = lr.predict(x_test)
actual = y_test

train_score_lr = lr.score(x_train, y_train)
test_score_lr = lr.score(x_test, y_test)

print("The train score for lr model is {}".format(train_score_lr))
print("The test score for lr model is {}".format(test_score_lr))

ridgeReg = Ridge(alpha=10)

ridgeReg.fit(x_train,y_train)

#train and test scorefor ridge regression
train_score_ridge = ridgeReg.score(x_train, y_train)
test_score_ridge = ridgeReg.score(x_test, y_test)

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))


# In[67]:


plt.figure(figsize = (10, 10))
plt.plot(features,ridgeReg.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 10$',zorder=7)
#plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$')
plt.plot(features,lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xticks(rotation = 90)
plt.legend()
plt.show()


# In[112]:


lasso = Lasso(alpha = 10)
lasso.fit(x_train,y_train)
train_score_ls =lasso.score(x_train,y_train)
test_score_ls =lasso.score(x_test,y_test)

print("The train score for ls model is {}".format(train_score_ls))
print("The test score for ls model is {}".format(test_score_ls))


# In[113]:


#Lasso Cross validation
lasso_cv = LassoCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10,100,1000], random_state=0).fit(x_train, y_train)


#score
print(lasso_cv.score(x_train, y_train))
print(lasso_cv.score(x_test, y_test))


# In[98]:


plt.figure(figsize = (10, 10))
#add plot for ridge regression
plt.plot(features,ridgeReg.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 10$',zorder=7)

#addd plot for lasso regression
plt.plot(lasso_cv.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'lasso; $\alpha = grid$')

#add plot for linear model
plt.plot(features,lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')

#rotate axis
plt.xticks(rotation = 90)
plt.legend()
plt.title("Comparison plot of Ridge, Lasso and Linear regression model")
plt.show()


# In[244]:


##Interpolating Average Transaction fees for ETH

df_interpol = pd.read_excel(r'C:\Users\tgome\Documents\Tomas Gomez Villa\2022-1 RSB\RGP\Raw_Data_Last_Version.xlsx', sheet_name='ETH_Ave_Tfees_Inter1')


# In[241]:


df_interpol.columns


# In[242]:


na_count=df_interpol.isna()
na_count.sum()


# In[243]:


series=df_interpol


# In[249]:


series_int=df_interpol['Transaction fees'].interpolate(method='spline',order=5)
plt.plot(df_interpol['DateTime'],series_int)
plt.show()


# In[251]:


series_int.to_csv(r'C:\Users\tgome\Documents\Tomas Gomez Villa\2022-1 RSB\RGP\new_export.csv')


# In[220]:


X_train = x_inter_data[0:23]
X_test = x_inter_data[23:]
len(X_train)


# In[212]:


def polynomial_fit(degree = 1):
  return np.poly1d(np.polyfit(X_train,X_test,degree))


# In[222]:


results = []
for i in range(1,len(X_train)-1):
  p = polynomial_fit(i)
  rmse = np.sqrt(mean_squared_error(X_test,p(X_train)))
  results.append({'degree':i,
    'rmse':rmse})
plt.scatter([x['degree'] for x in results],[x['rmse'] for x in results],label="RMSE")
plt.xlabel("Polynomial degree")
plt.legend()
plt.show()


# In[ ]:


## Interpreting interpolation results: To avoid overfitting a 4 degree interpolating equation will be computed in MS Excel according to the previous chart. Although these results seem to indicate that a 


# In[ ]:





# In[ ]:




