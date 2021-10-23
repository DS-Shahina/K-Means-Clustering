"Problem 4"

import pandas as pd

# Read data into Python
Tcc = pd.read_excel("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Dataset_Assignment Clustering/Telco_customer_churn.xlsx")
Tcc.head()
Tcc.shape
Tcc.columns.values
Tcc.dtypes
Tcc.info()

#  Delete column which is not necessary
# Only 11 columns has numeric values rest of all are categorical values
#let's remove columns which are not neended - we remove binary and categorical columns because it creates problem while clustering
Tcc1 = Tcc[['Number of Referrals','Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge','Total Charges','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']]
Tcc1.head()
Tcc1.shape
Tcc1.columns.values
Tcc1.dtypes
Tcc1.info()

#Exploratory Data Analysis
# Check the descriptive statistics of numeric variables
Tcc1.describe()
#1st moment Business Decision # Measures of Central Tendency / First moment business decision
Tcc1.mean()
Tcc1.median()
Tcc1.mode()

#2nd moment business decision # Measures of Dispersion / Second moment business decision
Tcc1.var() 
Tcc1.std()

#3rd moment Business Decision
Tcc1.skew()

#4th moment Business Decision
Tcc1.kurtosis()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np
import seaborn as sns
#Histogram
for i, predictor in enumerate(Tcc1):
    plt.figure(i)
    sns.histplot(data=Tcc1, x=predictor)
    
#boxplot    
for i, predictor in enumerate(Tcc1):
    plt.figure(i)
    sns.boxplot(data=Tcc1, x=predictor)

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)
 
# Normalized data frame (considering the numerical part of data)
Tcc1_norm = norm_func(Tcc1) # we take numeric columns, becuase that binary varibales create problem while clustering
Tcc1_norm.describe() # min=0, max=1
Tcc1_norm.info()

###################### Outlier Treatment #########
"Number of Referrals","Avg Monthly GB Download","Total Refunds","Total Extra Data Charges","Total Long Distance Charges","Total Revenue"
# so we have 6 variables which has outliers
# we leave some binary variable because that is categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)

# let's find outliers 
"Number of Referrals"
sns.boxplot(Tcc1['Number of Referrals']);plt.title('Boxplot');plt.show()

###################### Winsorization #####################################
import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Number of Referrals'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Number of Referrals']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Number of Referrals']);plt.title('Number of Referrals');plt.show()

#we see no outiers

"Avg Monthly GB Download"
sns.boxplot(Tcc1['Avg Monthly GB Download']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Avg Monthly GB Download'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Avg Monthly GB Download']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Avg Monthly GB Download']);plt.title('Avg Monthly GB Download');plt.show()

#we see no outiers

"Total Refunds"
sns.boxplot(Tcc1['Total Refunds']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Total Refunds'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Total Refunds']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Total Refunds']);plt.title('Total Refunds');plt.show()

#we see no outiers

"Total Extra Data Charges"
sns.boxplot(Tcc1['Total Extra Data Charges']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Total Extra Data Charges'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Total Extra Data Charges']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Total Extra Data Charges']);plt.title('Total Extra Data Charges');plt.show()

#we see no outiers

"Total Long Distance Charges"
sns.boxplot(Tcc1['Total Long Distance Charges']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Total Long Distance Charges'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Total Long Distance Charges']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Total Long Distance Charges']);plt.title('Total Long Distance Charges');plt.show()

#we see no outiers

"Total Revenue"
sns.boxplot(Tcc1['Total Revenue']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Total Revenue'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Tcc1_t = winsorizer.fit_transform(Tcc1[['Total Revenue']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Tcc1_t['Total Revenue']);plt.title('Total Revenue');plt.show()

#we see no outiers

#################### Missing Values Imputation ##################################
# check for count of NA'sin each column
Tcc1.isna().sum()

# there is no na values

################## Type casting###############################################
#Identify duplicates records in the data
duplicate = Tcc1.duplicated()
sum(duplicate)

#there is no duplicate values in the data

# Model Building 
# for creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # this is for Dendrogram
import gower #pip install gower in console window
a = gower.gower_matrix(Tcc1)
a
###### scree plot or elbow curve ############
from sklearn.cluster import KMeans
TWSS = []
k = list(range(2, 9)) #range is random
k

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(Tcc1_norm)
    TWSS.append(kmeans.inertia_) # total within sum of square (variance)
    
TWSS
# Scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(Tcc1_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
Tcc1['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
Tcc1 = Tcc1.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]] # we take numeric columns, becuase that binary varibales create problem while clustering
Tcc1.head()
Tcc1_norm.head()

Tcc1.iloc[:,1:].groupby(Tcc1.clust).mean()

Tcc1.to_csv("TelcoDataoutput1.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/K Means Clustering"
os.chdir(path) # current working directory

"Customer churn, also known as customer attrition, 
"occurs when customers stop doing business with a company or stop using a company’s services. 
"By being aware of and monitoring churn rate, companies are equipped to determine their customer retention success rates and identify strategies for improvement. 
"We will use a machine learning model to understand the precise customer behaviors and attributes., 
"Basically we are trying to find out the Customer churn pattern using K-Means clustering. 
"The primary objective is to identify Customer churn segments via K-Means clustering and design targeted marketing campaigns for each segment.

"In this K-Means clustering:-
"Customer Churn1 : 1st group (Highest number)
"Customer Churn2 : 3rd group (second Highest number)
"Customer Churn3 : 2nd group (third Highest number)
"Customer Churn4 : 0th group (fourth Highest number)

# now k = 5
# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(Tcc1_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
Tcc1['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership

Tcc1.iloc[:,1:].groupby(Tcc1.clust).mean()

Tcc1.to_csv("TelcoDataoutput2.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/K Means Clustering"
os.chdir(path) # current working directory

"In this K-Means clustering:-
"Customer Churn1 : 2nd group (Highest number)
"Customer Churn2 : 1st group (second Highest number)
"Customer Churn3 : 4th group (third Highest number)
"Customer Churn4 : 0th group (fourth Highest number)
"Customer Churn5 : 3rd group (fifth Highest number)

"6. Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided. "

"On the basis K-Means clustering We know more about Customer churn based on some varibales, 
"Basically we are trying to find out the Customer Churn using K-Means clustering. 
"The primary objective is to identify Customer Churn segments via K-Means clustering and design targeted marketing campaigns for each segment.
"so, because of K-Means clustering we can eassily figure out which group has the highest Customer Churn and which has lowest, and we can also see individual variables and know about Customer Churn"






















