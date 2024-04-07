#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[2]:


iris = load_iris()
X, y  = iris.data, iris.target


# In[3]:


pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)


# In[5]:


plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
plt.legend()
plt.xlabel("Temel Bileşen 1")
plt.ylabel("Temel Bileşen 2")
plt.title("Iris Veriseti")
plt.savefig('iris_pca.png')


# In[ ]:




