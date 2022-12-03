#-------------------------------------------------------------------------
# AUTHOR: Michael Phu
# FILENAME: clustering.py
# SPECIFICATION: This program tests the k-means algorithm at different sizes of k to 
# determine which value of 'k' maximizes the silhouette coefficient. It also serves to calculate the homogeniety score. 
# FOR: CS 4210- Assignment #5
# TIME SPENT: 1 Hour 30 Minutes 
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics
import csv 


df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library


# y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training
# assign your training data to X_training feature matrix
X_training = np.array(df.values)[:,:64] # Includes class 

#run kmeans testing different k values from 2 until 20 clusters
ks = [] 
coefficients = []
maxCoefficient = 0 
k_silhouette_coefficient = 2 
for k in range(2,21):
     ks.append(k)
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)

    
     # for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     # find which k maximizes the silhouette_coefficient
     coefficients.append(silhouette_score(X_training, kmeans.labels_))
     
     if maxCoefficient < coefficients[-1]:
          maxCoefficient = coefficients[-1]
          k_silhouette_coefficient = k

# Maximized Silhouette Coefficient for k-mean (2-20)
print("Maximized silhouette coefficient:",maxCoefficient,"(" + str(k_silhouette_coefficient) + "-clusters)")

#reading the test data (clusters) by using Pandas library
df = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
labels = np.array(df.values).reshape(1,len(df))[0]

# Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.scatter(ks,coefficients,alpha=0.5)
plt.xlabel('K')
plt.ylabel('Silhouette_Coefficient')
plt.title("K-Means Vs Silhouette_Coefficient")
plt.show()
