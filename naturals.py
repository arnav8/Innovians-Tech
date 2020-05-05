
# coding: utf-8

# In[1]:

#Unstructured Data
import cv2


# In[2]:

l = r'E:\ML_Codes\natural-images\images\car_0011.jpg'


# In[3]:

c = cv2.imread(l)


# In[5]:

cv2.imshow('My car', c)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:

#Create a array of labels of all the images
import os
image_name = os.listdir(r'E:\ML_Codes\natural-images\images')


# In[11]:

label = []
for i in image_name:
    if i. split("_")[0] == 'car':
        label.append(0) 
    if i. split("_")[0] == 'person':
        label.append(1)
    if i. split("_")[0] == 'motorbike':
        label.append(2)


# In[13]:

#Labels
Y = label


# In[15]:

loc = r'E:\ML_Codes\natural-images\images'


# In[21]:

features = []
for i in os.listdir(loc):
    x = os.path.join(loc,i)
    f = cv2.imread(x)
    rf = cv2.resize(f,(70,70))
    features.append(rf)


# In[25]:

X = features


# In[22]:

import numpy as np
np.array(features).shape


# In[24]:

d = features[20]
cv2.imshow('image',d)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:

image_name
k = 'car_0000.jpg'
k.split('_')[0]


# In[ ]:

#Create a feature matrix using all the image

