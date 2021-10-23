"Problem 3"


import pandas as pd

# Read data into Python
Insurance_Data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Unsupervised-K-Means Clustering/Datasets_Kmeans/Insurance Dataset.csv")
Insurance_Data.head()
Insurance_Data.shape
Insurance_Data.columns.values
Insurance_Data.dtypes
Insurance_Data.info()


#Exploratory Data Analysis
# Check the descriptive statistics of numeric variables
Insurance_Data.describe()
#1st moment Business Decision # Measures of Central Tendency / First moment business decision
Insurance_Data.mean()
Insurance_Data.median()
Insurance_Data.mode()

#2nd moment business decision # Measures of Dispersion / Second moment business decision
Insurance_Data.var() 
Insurance_Data.std()

#3rd moment Business Decision
Insurance_Data.skew()

#4th moment Business Decision
Insurance_Data.kurtosis()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np
import seaborn as sns
#Histogram
for i, predictor in enumerate(Insurance_Data):
    plt.figure(i)
    sns.histplot(data=Insurance_Data, x=predictor)
    
#boxplot    
for i, predictor in enumerate(Insurance_Data):
    plt.figure(i)
    sns.boxplot(data=Insurance_Data, x=predictor)
    
#Normal Quantile-Quantile Plot

import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
#Premiums Paid
stats.probplot(Insurance_Data['Premiums Paid'], dist='norm',plot=pylab) #pylab is visual representation
#transformation to make Premiums Paid variable normal
import numpy as np
#Age
stats.probplot(Insurance_Data['Age'], dist='norm',plot=pylab)
#Days to Renew
stats.probplot(Insurance_Data['Days to Renew'], dist='norm',plot=pylab)
stats.probplot(np.sqrt(Insurance_Data['Days to Renew']),dist="norm",plot=pylab)
#Claims made
stats.probplot(Insurance_Data['Claims made'], dist='norm',plot=pylab)
stats.probplot(np.log(Insurance_Data['Claims made']),dist="norm",plot=pylab)
#Income
stats.probplot(Insurance_Data['Income'], dist='norm',plot=pylab)
stats.probplot(np.sqrt(Insurance_Data['Income']),dist="norm",plot=pylab)
    

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)
 
# Normalized data frame
Insurance_Data_norm = norm_func(Insurance_Data) 
Insurance_Data_norm.describe() # min=0, max=1
Insurance_Data_norm.info()


###################### Outlier Treatment #########
"Premiums Paid","Claims made"

# let's find outliers 
"Premiums Paid"
sns.boxplot(Insurance_Data['Premiums Paid']);plt.title('Boxplot');plt.show()

###################### Winsorization #####################################
import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Premiums Paid'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Insurance_Data_t = winsorizer.fit_transform(Insurance_Data[['Premiums Paid']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Insurance_Data_t['Premiums Paid']);plt.title('Premiums Paid');plt.show()

#we see no outiers

"Claims made"
sns.boxplot(Insurance_Data['Claims made']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Claims made'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Insurance_Data_t = winsorizer.fit_transform(Insurance_Data[['Claims made']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Insurance_Data_t['Claims made']);plt.title('Claims made');plt.show()

#we see no outiers

#################### Missing Values Imputation ##################################
# check for count of NA'sin each column
Insurance_Data.isna().sum()

# there is no na values

################## Type casting###############################################
#Identify duplicates records in the data
duplicate = Insurance_Data.duplicated()
sum(duplicate)

#there is no duplicate values in the data

# Model Building 
###### scree plot or elbow curve ############
from sklearn.cluster import KMeans
TWSS = []
k = list(range(2, 11)) #range is random
k

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(Insurance_Data_norm)
    TWSS.append(kmeans.inertia_) # total within sum of square (variance)
    
TWSS
# Scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(Insurance_Data_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
Insurance_Data['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
Insurance_Data1 = Insurance_Data.iloc[:,[5,0,1,2,3,4]]
Insurance_Data1.head()
Insurance_Data_norm.head()

Insurance_Data1.iloc[:].groupby(Insurance_Data1.clust).mean()

Insurance_Data1.to_csv("Insurance_Dataoutput1.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/K Means Clustering"
os.chdir(path) # current working directory

"We are trying to learn more about Person based on the Age, Days to Renew, Claims made, Income, Premiums Paid. 
"Basically we are trying to find out the Person Pattern using K-Means clustering. 
"The primary objective is to identify Person Pattern segments via K-Means clustering and design targeted marketing campaigns for each segment.

"In this K-Means clustering:-
"Person Pattern1 : 1st group (Highest number)
"Person Pattern2 : 2nd group (second Highest number)
"Person Pattern3 : 0th group (third Highest number)

# now k = 4
# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(Insurance_Data_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
Insurance_Data['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
Insurance_Data2 = Insurance_Data.iloc[:,[5,0,1,2,3,4]]
Insurance_Data2.head()
Insurance_Data_norm.head()

Insurance_Data2.iloc[:].groupby(Insurance_Data2.clust).mean()

Insurance_Data2.to_csv("Insurance_Dataoutput2.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/K Means Clustering"
os.chdir(path) # current working directory

"In this K-Means clustering:-
"Person Pattern1 : 0th group (Highest number)
"Person Pattern2 : 1st group (second Highest number)
"Person Pattern3 : 3rd group (third Highest number)
"Person Pattern4 : 2nd group (fourth Highest number)

"6. Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided. "

"We are trying to learn more about Person based on the Age, Days to Renew, Claims made, Income, Premiums Paid. 
"Basically we are trying to find out the Person Pattern using K-Means clustering. 
"The primary objective is to identify Person Pattern segments via K-Means clustering and design targeted marketing campaigns for each segment.
"so, because of KMeans clustering we can eassily figure out which group has the highest and which has lowest, and we can also see individual variables and know about Persons Pattern "
