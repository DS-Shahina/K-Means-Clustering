#K-means clustering is applied for large dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#Generaing random uniform numbers
# generated 50 random values from uniformely generated data and the range of uniformely generated data is 0 to 1
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=['A','B']) # empty dataframe
df_xy.A = X # STORE VALUE IN 'A' COLUMN
df_xy.B = Y # STORE VALUE IN 'B' COLUMN

df_xy.plot(x="A", y="B", kind="scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "A", y ="B", c = model1.labels_, kind="scatter", s = 10, cmap=plt.cm.coolwarm)
# s- Scale on y axis

# Kmeans on University Data set 
univ1 = pd.read_excel("C:/Users/Admin/Downloads/University_Clustering.xlsx")
univ1.describe()

univ = univ1.drop(["State"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(univ.iloc[:,1:])

###### scree plot or elbow curve ############

TWSS = []
k = list(range(2, 9))
k

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_) # total within sum of square (variance), inertia means standard deviation - total within sum of square.
    
TWSS
# Scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
univ['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership

univ.head()
df_norm.head()

univ = univ.iloc[:,[7,0,1,2,3,4,5,6]]
univ.head()

univ.iloc[:, 2:8].groupby(univ.clust).mean()

univ.to_csv("Kmeans_universityoutput.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Unsupervised-K-Means Clustering"
os.chdir(path) # current working directory


