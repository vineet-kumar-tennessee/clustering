# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:03:55 2024

@author: khull
"""
#pip install xgboost
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
import pickle

# %matplotlib inline
from PIL import Image

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#from xgboost import XGBClassifier
from sklearn.cluster import KMeans

d=pd.read_csv(r"data location\Mall_Customers.csv")

x = d.iloc[:,[3,4]].values

#wcss=[]

#for i in range(1,11):
#    k=KMeans(n_clusters=i, random_state=0)
#    k.fit(x)
#    wcss.append(k.inertia_)
#plt.plot(range(1,11),wcss)

k=KMeans(n_clusters=5, init='k-means++',random_state=0)
X=x
y_kmeans=k.fit_predict(x)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(k.cluster_centers_[:, 0], k.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
#plt.show()

print('wd:', os.getcwd())


with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(k,file)