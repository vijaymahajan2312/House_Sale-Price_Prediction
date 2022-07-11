#!/usr/bin/env python
# coding: utf-8

# # Read Dataset

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
train = pd.read_csv("C:/Documents/DataScience/1/training_set.csv")
train.head()


# In[2]:


train.info()


# # Dealing with Missing Data

# In[3]:


train.Alley = train.Alley.fillna("No_Alley")


# In[4]:


train.FireplaceQu = train.FireplaceQu.fillna("No_Fireplace")


# In[5]:


train.PoolQC = train.PoolQC.fillna("No_Pool")


# In[6]:


train.Fence = train.Fence.fillna("No_Fence")


# In[7]:


train.MiscFeature = train.MiscFeature.fillna("None")


# In[8]:


for i in train.columns:
    if(train[i].dtype=='object'):
        x = train[i].mode()[0]
        train[i] = train[i].fillna(x)
    else:
        x = train[i].mean()
        train[i] = train[i].fillna(x)


# In[9]:


train.info()


# # Define our Y column and temp X column

# In[10]:


Y = train[["SalePrice"]]
X = train.drop(labels=["SalePrice","Id"],axis=1)


# # Exploratory Data Analysis

# In[11]:


train.corr()[["SalePrice"]].sort_values(by="SalePrice")


# In[12]:


train.corr()[["SalePrice"]].sort_values(by="SalePrice").tail(20).index


# In[13]:


from warnings import filterwarnings
filterwarnings("ignore")
x =1
plt.figure(figsize=(10,55))

for i in train.columns:
    if(train[i].dtype=='object'):
        plt.subplot(41,2,x)
        sb.boxplot(train.SalePrice,train[i])
        x = x+1
    else:
        plt.subplot(41,2,x)
        sb.scatterplot(train.SalePrice,train[i])
        x=x+1


# In[14]:


plt.figure(figsize=(99,81))
sb.heatmap(train.corr(),annot=True,cmap='viridis')


# In[15]:


sb.lineplot(train.SalePrice,train.OverallQual)


# In[16]:


sb.regplot(train.SalePrice,train.OverallQual)


# In[17]:


sb.regplot(train.FullBath,train.SalePrice)


# In[18]:


sb.regplot(train.YearBuilt,train.SalePrice)


# In[19]:


sb.regplot(train.GrLivArea,train.SalePrice)


# In[20]:


sb.regplot(train.GarageCars,train.SalePrice)


# In[21]:


imp_con_cols = list(train.corr()['SalePrice'].sort_values().tail(20).index)
imp_con_cols.remove("SalePrice")


# In[22]:


def ANOVA(df,cat,con):
    from pandas import DataFrame
    from statsmodels.api import OLS
    from statsmodels.formula.api import ols
    rel = con + " ~ " + cat
    model = ols(rel,df).fit()
    from statsmodels.stats.anova import anova_lm
    anova_results = anova_lm(model)
    Q = DataFrame(anova_results)
    a = Q['PR(>F)'][cat]
    return round(a,3)
cat = []
con = []
for i in X.columns:
    if(X[i].dtypes == "object"):
        cat.append(i)
    else:
        con.append(i)


# In[23]:


imp_cat_cols = []
for i in cat:
    pval = ANOVA(train,i,'SalePrice')
    print("SalePrice vs",i,ANOVA(train,i,'SalePrice'))
    if(pval < 0.05):
        imp_cat_cols.append(i)


# In[24]:


X[imp_cat_cols]


# In[25]:


X[imp_con_cols].skew()


# In[26]:


sb.distplot(X[imp_con_cols].skew())


# (X[imp_con_cols].apply(np.log)).skew()

# In[27]:


X[imp_con_cols].apply(np.log)


# In[28]:


X[imp_con_cols].skew()


# In[29]:


imp_cols =[]
imp_cols.extend(imp_con_cols)
imp_cols.extend(imp_cat_cols)


# In[30]:


X = X[imp_cols]


# # Preprocessing

# In[31]:


cat =[]
con=[]
for i in X.columns:
    if(X[i].dtypes=='object'):
        cat.append(i)
    else:
        con.append(i)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X1 = pd.get_dummies(X[cat])
X2= pd.DataFrame(ss.fit_transform(X[con]), columns=con)
Xnew = X2.join(X1)
Xnew


# # OLS(Linear model)

# In[32]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(Xnew,Y, test_size=0.2, random_state=31)

from statsmodels.api import OLS,add_constant
xconst = add_constant(xtrain)
ols = OLS(ytrain,xconst)
model = ols.fit()
model.summary()


# In[33]:


Q = pd.DataFrame(model.pvalues, columns=["Pval"]) #backward elemination OLS model
col_to_drop = list(Q.sort_values(by=['Pval']).tail(1).index)
print(col_to_drop)

Xnew = Xnew.drop(labels=col_to_drop,axis=1)
xtrain,xtest,ytrain,ytest = train_test_split(Xnew,Y,test_size=0.2, random_state=21)
from statsmodels.api import OLS,add_constant
xconst = add_constant(xtrain)
ols = OLS(ytrain,xconst)
model = ols.fit()
model.summary()


# In[34]:


Q = pd.DataFrame(model.pvalues, columns=["Pval"]) #backward elemination OLS model
col_to_drop = list(Q.sort_values(by=['Pval']).tail(1).index)
print(col_to_drop)

Xnew = Xnew.drop(labels=col_to_drop,axis=1)
xtrain,xtest,ytrain,ytest = train_test_split(Xnew,Y,test_size=0.2, random_state=21)
from statsmodels.api import OLS,add_constant
xconst = add_constant(xtrain)
ols = OLS(ytrain,xconst)
model = ols.fit()
model.summary()


# In[35]:


Q = pd.DataFrame(model.pvalues, columns=["Pval"]) #backward elemination OLS model
col_to_drop = list(Q.sort_values(by=['Pval']).tail(1).index)
print(col_to_drop)

Xnew = Xnew.drop(labels=col_to_drop,axis=1)
xtrain,xtest,ytrain,ytest = train_test_split(Xnew,Y,test_size=0.2, random_state=21)
from statsmodels.api import OLS,add_constant
xconst = add_constant(xtrain)
ols = OLS(ytrain,xconst)
model = ols.fit()
model.summary()


# In[36]:


Q = pd.DataFrame(model.pvalues, columns=["Pval"]) #backward elemination OLS model
col_to_drop = list(Q.sort_values(by=['Pval']).tail(1).index)
print(col_to_drop)   # Remove unncessary columns on the basis of pval

Xnew = Xnew.drop(labels=col_to_drop,axis=1)
xtrain,xtest,ytrain,ytest = train_test_split(Xnew,Y,test_size=0.2, random_state=21)
from statsmodels.api import OLS,add_constant
xconst = add_constant(xtrain)
ols = OLS(ytrain,xconst)
model = ols.fit()
model.summary()


# # Model

# In[37]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
model = lr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
tr_error = mean_absolute_error(ytrain,tr_pred)
ts_error = mean_absolute_error(ytest,ts_pred)


# In[38]:


tr_error


# In[39]:


ts_error


# # Regularize

# In[40]:


Q = list(Xnew.columns)
Q.extend(con)


# In[41]:


duplicates =[]
for i in Q:
    if(Q.count(i)>1 and duplicates.count(i)==0):
        duplicates.append(i)


# In[42]:


Xnew[duplicates].corr()


# In[43]:


grid =[]
w=0.9
for i in range(0,10,1):
    grid.append(w)
    w = w+0.001

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
ls = Lasso()
tg = {"alpha": grid}
cv = GridSearchCV(ls,tg,scoring="neg_mean_absolute_error", cv=3)
cvmodel = cv.fit(xtrain,ytrain)
cvmodel.best_params_


# In[44]:


ls = Lasso(alpha=0.909)
model = ls.fit(Xnew,Y)


# In[45]:


Xnew.shape


# # Testing Data 

# In[46]:


test = pd.read_csv("C:/Documents/DataScience/1/testing_set_House.csv")
test.head()


# In[47]:


X2 = test.drop(labels='Id', axis=1)


# In[48]:


X2.info()


# # Dealing with Missing Data

# In[49]:


train.Alley = train.Alley.fillna("No_Alley")
train.Alley = train.Alley.fillna("No_Alley")
train.FireplaceQu = train.FireplaceQu.fillna("No_Fireplace")
train.PoolQC = train.PoolQC.fillna("No_Pool")
train.Fence = train.Fence.fillna("No_Fence")
train.MiscFeature = train.MiscFeature.fillna("None")

for i in X2.columns:
    if(X2[i].dtype=='object'):
        x = X2[i].mode()[0]
        X2[i] = X2[i].fillna(x)
    else:
        x = X2[i].mean()
        X2[i] = X2[i].fillna(x)


# In[50]:


X2.info()


# In[51]:


cat =[]
con=[]
for i in X2.columns:
    if(X2[i].dtypes=='object'):
        cat.append(i)
    else:
        con.append(i)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X1 = pd.get_dummies(X2[cat])
X2= pd.DataFrame(ss.fit_transform(X2[con]), columns=con)
Xnew2 = X2.join(X1)
Xnew2


# In[52]:


Xnew.shape


# In[53]:


Xnew2.shape


# In[54]:


Q =['PoolQC_Fa', 'Heating_OthW', 'PoolQC_No_Pool', 'Exterior1st_Stone', 'Heating_Floor', 'Exterior1st_ImStucc', 'Fence_No_Fence', 'Exterior2nd_Other', 'FireplaceQu_No_Fireplace', 'Alley_No_Alley', 'RoofMatl_Metal', 'Condition2_RRNn', 'RoofMatl_Roll', 'MiscFeature_None', 'Condition2_RRAe', 'HouseStyle_2.5Fin', 'MiscFeature_TenC', 'GarageQual_Ex', 'RoofMatl_Membran', 'Electrical_Mix', 'Condition2_RRAn']


# In[55]:


for i in Q:
    Xnew2[i]=0


# In[56]:


finalX = Xnew2[Xnew.columns]


# In[57]:


pred = model.predict(finalX)


# In[58]:


FinalDF = pd.DataFrame(test.Id)
FinalDF["SalePrice"] = pred


# In[59]:


FinalDF


# In[60]:


FinalDF.to_csv("C:/Documents/house_price_ans.csv")


# In[ ]:




