
# coding: utf-8

# In[1]:

import pandas as pd
mall = pd.read_csv(r'E:\ML_Codes\mall-customers\Mall_Customers.csv')


# In[2]:

import numpy as np
import matplotlib.pyplot as plt


# In[3]:

mall.head()


# In[4]:

mall.info()


# In[5]:

X = mall.loc[:,['Annual Income (k$)','Spending Score (1-100)']]


# In[6]:

X


# In[7]:

from sklearn.cluster import KMeans
loss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    loss.append(kmeans.inertia_)


# In[9]:

plt.plot(range(1,11),loss)
plt.show()


# In[11]:

kmeansmodel = KMeans(n_clusters = 5)
kmeansmodel.fit(X)
ykmeans = kmeansmodel.predict(X)


# In[13]:

ykmeans


# In[14]:

X.loc[0]


# In[15]:

#Earning high but spensing less
#Average in terms of earning and spending
#Earning high and spending high
#Earning less ans spensing high
#Earning less, spending less


# In[33]:

plt.scatter(X[ykmeans == 0].iloc[:,0] , X[ykmeans == 0].iloc[:,1], c= 'r',label='Cluster 1')
plt.scatter(X[ykmeans == 1].iloc[:,0] , X[ykmeans == 1].iloc[:,1], c= 'b', label='Cluster 2')
plt.scatter(X[ykmeans == 2].iloc[:,0] , X[ykmeans == 2].iloc[:,1], c= 'g', label='Cluster 3')
plt.scatter(X[ykmeans == 3].iloc[:,0] , X[ykmeans == 3].iloc[:,1], c= 'cyan', label='Cluster 4')
plt.scatter(X[ykmeans == 4].iloc[:,0] , X[ykmeans == 4].iloc[:,1], c= 'magenta', label='Cluster 5')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()


# In[42]:

#two coordinate
X
X[ykmeans == 0].iloc[:,1]


# In[ ]:



