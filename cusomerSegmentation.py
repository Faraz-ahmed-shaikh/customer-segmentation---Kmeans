# Importing Liabraries and Loading Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
os.environ["OMP_NUM_THREADS"] = "1"
df =  pd.read_csv("Mall_Customers.csv")

# Exploratory Data Analysis
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

# Data Vizualization 
gender = df.groupby('Gender')['Age'].count()
plt.pie(gender.values,labels=gender.index, autopct="%1.f%%", wedgeprops={"linewidth":3,"edgecolor":"white"},startangle=90, colors=["pink","blue"])
plt.show()

plt.hist(df['Age'], bins=4, edgecolor="white")
plt.grid(True,alpha=0.2)
plt.xlabel("Age Group")
plt.ylabel("Np. of People")
plt.title("Age Distribution")
plt.show()

# Elbow Methods - for deciding suitable number of K / clusters
X = df[["Annual Income (k$)","Spending Score (1-100)"]]
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('SSE')
plt.show()

# Customer Segmentation
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(8,6))
plt.scatter(X.iloc[y_kmeans == 0, 0], X.iloc[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X.iloc[y_kmeans == 1, 0], X.iloc[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X.iloc[y_kmeans == 2, 0], X.iloc[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X.iloc[y_kmeans == 3, 0], X.iloc[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X.iloc[y_kmeans == 4, 0], X.iloc[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

# Centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', marker='o', label='Centroids')

plt.title('Customer Segmentation - ML Project')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
