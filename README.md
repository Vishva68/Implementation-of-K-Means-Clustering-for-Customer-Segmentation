# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries, load the Mall_Customers.csv dataset, and check for missing values.
2. Use the Elbow Method by calculating WCSS values for different cluster counts and plot the graph.
3. Create the K-Means clustering model with the optimal number of clusters and fit it to the dataset.
4. Predict cluster labels for customers and visualize customer segments using a scatter plot.

## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by  :VISHVANANDH N 
RegisterNumber:212224240186
```
```
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=0,
        n_init=10
    )

    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')

plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()

km = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=0,
    n_init=10
)

km.fit(data.iloc[:, 3:])

y_pred = km.predict(data.iloc[:, 3:])

print(y_pred)

data["cluster"] = y_pred

df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

plt.figure(figsize=(10,6))

plt.scatter(
    df0["Annual Income (k$)"],
    df0["Spending Score (1-100)"],
    c="red",
    label="cluster0"
)

plt.scatter(
    df1["Annual Income (k$)"],
    df1["Spending Score (1-100)"],
    c="black",
    label="cluster1"
)

plt.scatter(
    df2["Annual Income (k$)"],
    df2["Spending Score (1-100)"],
    c="blue",
    label="cluster2"
)

plt.scatter(
    df3["Annual Income (k$)"],
    df3["Spending Score (1-100)"],
    c="green",
    label="cluster3"
)

plt.scatter(
    df4["Annual Income (k$)"],
    df4["Spending Score (1-100)"],
    c="magenta",
    label="cluster4"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")

plt.legend()
plt.show()
```

## Output:
<img width="870" height="480" alt="Screenshot 2026-05-25 110544" src="https://github.com/user-attachments/assets/08c8794c-e15b-4e64-a0bc-b31d74202979" />
<img width="1017" height="528" alt="Screenshot 2026-05-25 110554" src="https://github.com/user-attachments/assets/4179818e-5b9d-4cfe-bfc6-1463f626baf8" />
<img width="925" height="139" alt="Screenshot 2026-05-25 110606" src="https://github.com/user-attachments/assets/0f0c68e4-36a1-49cc-bd13-c34f754da90a" />
<img width="1078" height="554" alt="Screenshot 2026-05-25 110851" src="https://github.com/user-attachments/assets/9a1361ef-3101-4a1e-bfaa-9b05448401a3" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
