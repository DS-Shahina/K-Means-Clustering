"Problem 2"

import pandas as pd

# Read data into Python
crime_data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Dataset_Assignment Clustering/crime_data.csv")

#Exploratory Data Analysis
#Measures of Central Tendency / First moment business decision
#Murder
crime_data.Murder.mean() # '.' is used to refer to the variables within object
crime_data.Murder.median()
crime_data.Murder.mode()

#Assault
crime_data.Assault.mean() 
crime_data.Assault.median()
crime_data.Assault.mode()

#UrbanPop
crime_data.UrbanPop.mean() 
crime_data.UrbanPop.median()
crime_data.UrbanPop.mode()

#Rape
crime_data.Rape.mean() 
crime_data.Rape.median()
crime_data.Rape.mode()

# Measures of Dispersion / Second moment business decision
#Murder
crime_data.Murder.var() # variance
crime_data.Murder.std() #standard deviation
range = max(crime_data.Murder) - min(crime_data.Murder) # range
range

#Assault
crime_data.Assault.var() # variance
crime_data.Assault.std() #standard deviation
range = max(crime_data.Assault) - min(crime_data.Assault) # range
range

#UrbanPop
crime_data.UrbanPop.var() # variance
crime_data.UrbanPop.std() #standard deviation
range = max(crime_data.UrbanPop) - min(crime_data.UrbanPop) # range
range

#Rape
crime_data.Rape.var() # variance
crime_data.Rape.std() #standard deviation
range = max(crime_data.Rape) - min(crime_data.Rape) # range
range

# Third moment business decision
crime_data.Murder.skew() # +ve skew , right skew
crime_data.Assault.skew() # +ve skew , right skew
crime_data.UrbanPop.skew() # -ve skew , left skew
crime_data.Rape.skew() # +ve skew , right skew


#Fourth moment business decision
crime_data.Murder.kurt() # platykurtic
crime_data.Assault.kurt() #platykurtic 
crime_data.UrbanPop.kurt() # platykurtic
crime_data.Rape.kurt() #platykurtic

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np


plt.hist(crime_data.Murder) #histogram
plt.hist(crime_data.Assault)
plt.hist(crime_data.UrbanPop)
plt.hist(crime_data.Rape)

#boxplot
plt.boxplot(crime_data.Murder,1,vert=False) #it has no outliers 
plt.boxplot(crime_data.Assault,1,vert=False) #it has no outliers
plt.boxplot(crime_data.UrbanPop,1,vert=False) # no outliers 
plt.boxplot(crime_data.Rape,1,vert=False) # it has outliers and it is right skew

#Normal Quantile-Quantile Plot

import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
#Murder
stats.probplot(crime_data.Murder, dist='norm',plot=pylab) #pylab is visual representation
#transformation to make workex variable normal
import numpy as np
#Assault
stats.probplot(crime_data.Assault, dist='norm',plot=pylab)
#UrbanPop
stats.probplot(crime_data.UrbanPop, dist='norm',plot=pylab)
#Rape
stats.probplot(crime_data.Rape, dist='norm',plot=pylab)
stats.probplot(np.log(crime_data.Rape),dist="norm",plot=pylab)

###### Data Preprocessing########################################

## import packages
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

crime_data.isna().sum()
crime_data.describe() # f na values then count will decrease
crime_data.info() #data types , #object - categorical data

##################  creating Dummy variables using dummies ###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create dummy variables on categorcal columns

crime_data.info()
crime_data.head()
for col in crime_data:
    print(crime_data[col].unique())
crime_data['States'].unique()
len(crime_data['States'].unique())

# let's have a look at how many labels each variable has

for col in crime_data.columns:
    print(col, ': ', len(crime_data[col].unique()), ' labels')
crime_data.shape    

# let's examine how many columns we will obtain after one hot encoding these variables
pd.get_dummies(crime_data).shape
#We can observe that from with just 5 categorical features we are getting 53 features with the help of one hot encoding.
# let's find the top 10 most frequent categories for the variable States

crime_data.States.value_counts().sort_values(ascending=False).head(20)

# let's make a list with the most frequent categories of the variable

top_10_labels = [y for y in crime_data.States.value_counts().sort_values(ascending=False).head(10).index]
top_10_labels

# get whole set of dummy variables, for all the categorical variables

def one_hot_encoding_top_x(crime_data, variable, top_x_labels):
    # function to create the dummy variables for the most frequent labels
    # we can vary the number of most frequent labels that we encode
    
    for label in top_x_labels:
        crime_data[variable+'_'+label] = np.where(crime_data[variable]==label, 1, 0)

crime_data = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Dataset_Assignment Clustering/crime_data.csv")
# encode States into the 10 most frequent categories
one_hot_encoding_top_x(crime_data, "States", top_10_labels)
crime_data.head()
crime_data.info()

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)
 
# Normalized data frame (considering the numerical part of data)
crime_data_norm = norm_func(crime_data.iloc[:,[1,2,3,4]]) # we take numeric columns, becuase that binary varibales create problem while clustering
crime_data_norm.describe() # min=0, max=1
crime_data_norm.info()

###################### Outlier Treatment #########
# so we have 1 variables which has outliers
# we leave some binary variable because that is categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)
crime_data.dtypes
crime_data.isna().sum()

# let's find outliers 
"Rape"
sns.boxplot(crime_data.Rape);plt.title('Boxplot');plt.show()

###################### Winsorization #####################################
import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Rape'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
crime_data_t = winsorizer.fit_transform(crime_data[['Rape']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(crime_data_t.Rape);plt.title('Rape');plt.show()

#we see no outiers

#################### Missing Values Imputation ##################################
# check for count of NA'sin each column
crime_data.isna().sum()

# there is no na values

################## Type casting###############################################
#Identify duplicates records in the data
duplicate = crime_data.duplicated()
sum(duplicate)

#there is no duplicate values in the data

# Model Building 
###### scree plot or elbow curve ############
from sklearn.cluster import KMeans
TWSS = []
k = list(range(2, 9))
k

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crime_data_norm)
    TWSS.append(kmeans.inertia_) # total within sum of square (variance)
    
TWSS
# Scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(crime_data_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
crime_data['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
crime_data = crime_data.iloc[:,[15,1,2,3,4]]
crime_data.head()
crime_data_norm.head()

crime_data.iloc[:, 1:].groupby(crime_data.clust).mean()

crime_data.to_csv("crime_dataoutput1.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/K Means Clustering"
os.chdir(path) # current working directory

"We are trying to learn more about Crime rate based on the states of U.S, 
"Basically we are trying to find out the crime rate using K-Means clustering. 
"The primary objective is to identify Crime rates segments via K-Means clustering and design targeted marketing campaigns for each segment.

"In this K-Means clustering:-
"Crime rate1 : 0th group (Highest number)
"Crime rate2 : 1st group (second Highest number)
"Crime rate3 : 2nd group (third Highest number)

# now k = 4
# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(crime_data_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
crime_data['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
crime_data1 = crime_data.iloc[:,[15,1,2,3,4]]
crime_data1.head()
crime_data_norm.head()

crime_data1.iloc[:, 1:].groupby(crime_data1.clust).mean()

crime_data1.to_csv("crime_dataoutput2.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/K Means Clustering"
os.chdir(path) # current working directory

"In this K-Means clustering:-
"Crime rate1 : 2nd group (Highest number)
"Crime rate2 : 1st group (second Highest number)
"Crime rate3 : 0th group (third Highest number)
"Crime rate4 : 3rd group (fourth Highest number)

"6. Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided. "

"On the basis K-Means clustering We know more about Crime rate based on the states of U.S, 
"Basically we are trying to find out the crime rate using KMeans clustering. 
"The primary objective is to identify Crime rates segments via KMeans clustering and design targeted marketing campaigns for each segment.
"so, because of KMeans clustering we can eassily figure out which group has the highest crime rate and which has lowest, and we can also see individual variables and know about crime rate "
