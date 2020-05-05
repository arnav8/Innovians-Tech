
# coding: utf-8

# In[9]:

import pandas as pd
import numpy as np


# In[22]:

digit = pd.read_csv(r'E:\ML_Codes\all\train.csv')
digit_test = pd.read_csv(r'E:\ML_Codes\all\test.csv').values


# In[11]:

digit
X = digit.drop(['label'],axis=1).values
Y = digit['label']


# In[28]:

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.15)


# In[29]:

from sklearn.neighbors import KNeighborsClassifier
kmodel = KNeighborsClassifier(n_neighbors=3)


# In[30]:

kmodel.fit(xtrain,ytrain)


# In[31]:

#Training Accuracy
kmodel.score(xtrain,ytrain)


# In[33]:

#Testing Accuracy
kmodel.score(xtest,ytest)


# In[34]:

from sklearn.tree import DecisionTreeClassifier
dmodel = DecisionTreeClassifier()


# In[ ]:

dmodel.fit(xtrain,ytrain)


# In[ ]:

#Training accuracy using Decision Tree
dmodel.score(xtrain,ytrain)


# In[ ]:

#Testing accuracy using Decision Tree\
dmodel.score(xtest,ytest)


# In[35]:

from sklearn.ensemble import RandomForestClassifier
rmodel = RandomForestClassifier(n_estimators = 50)


# In[ ]:

rmodel.fit(xtrain,ytrain)


# In[ ]:

#Training accuracy
rmodel.score(xtrain,ytrain)


# In[ ]:

#testing accuracy
rmodel.score(xtest,ytest)


# In[18]:

i = X[40].reshape(28,28)
i.shape


# In[21]:

print(Y[40])
import matplotlib.pyplot as plt
plt.imshow(i)
plt.show()


# In[25]:

d = digit_test[50].reshape(28,28)
plt.imshow(d)
plt.show()


# In[ ]:



