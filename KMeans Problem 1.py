"Problem 1"

import pandas as pd

# Read data into Python
EastWestAirlines = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Dataset_Assignment Clustering/EastWestAirlines.csv")

#Exploratory Data Analysis
#Measures of Central Tendency / First moment business decision
#Balance
EastWestAirlines.Balance.mean() # '.' is used to refer to the variables within object
EastWestAirlines.Balance.median()
EastWestAirlines.Balance.mode()

#Qual_miles
EastWestAirlines.Qual_miles.mean() 
EastWestAirlines.Qual_miles.median()
EastWestAirlines.Qual_miles.mode()

#cc1_miles
EastWestAirlines.cc1_miles.mean() 
EastWestAirlines.cc1_miles.median()
EastWestAirlines.cc1_miles.mode()

#Bonus_miles
EastWestAirlines.Bonus_miles.mean() 
EastWestAirlines.Bonus_miles.median()
EastWestAirlines.Bonus_miles.mode()

#Bonus_trans
EastWestAirlines.Bonus_trans.mean() 
EastWestAirlines.Bonus_trans.median()
EastWestAirlines.Bonus_trans.mode()

#Flight_miles_12mo
EastWestAirlines.Flight_miles_12mo.mean() 
EastWestAirlines.Flight_miles_12mo.median()
EastWestAirlines.Flight_miles_12mo.mode()

#Flight_trans_12
EastWestAirlines.Flight_trans_12.mean() 
EastWestAirlines.Flight_trans_12.median()
EastWestAirlines.Flight_trans_12.mode()

#Days_since_enroll
EastWestAirlines.Days_since_enroll.mean() 
EastWestAirlines.Days_since_enroll.median()
EastWestAirlines.Days_since_enroll.mode()

# Measures of Dispersion / Second moment business decision
#Balance
EastWestAirlines.Balance.var() # variance
EastWestAirlines.Balance.std() #standard deviation
range = max(EastWestAirlines.Balance) - min(EastWestAirlines.Balance) # range
range

#Qual_miles
EastWestAirlines.Qual_miles.var() # variance
EastWestAirlines.Qual_miles.std() #standard deviation
range = max(EastWestAirlines.Qual_miles) - min(EastWestAirlines.Qual_miles) # range
range

#cc1_miles
EastWestAirlines.cc1_miles.var() # variance
EastWestAirlines.cc1_miles.std() #standard deviation
range = max(EastWestAirlines.cc1_miles) - min(EastWestAirlines.cc1_miles) # range
range

#Bonus_miles
EastWestAirlines.Bonus_miles.var() # variance
EastWestAirlines.Bonus_miles.std() #standard deviation
range = max(EastWestAirlines.Bonus_miles) - min(EastWestAirlines.Bonus_miles) # range
range

#Bonus_trans
EastWestAirlines.Bonus_trans.var() # variance
EastWestAirlines.Bonus_trans.std() #standard deviation
range = max(EastWestAirlines.Bonus_trans) - min(EastWestAirlines.Bonus_trans) # range
range

#Flight_miles_12mo
EastWestAirlines.Flight_miles_12mo.var() # variance
EastWestAirlines.Flight_miles_12mo.std() #standard deviation
range = max(EastWestAirlines.Flight_miles_12mo) - min(EastWestAirlines.Flight_miles_12mo) # range
range

#Flight_trans_12
EastWestAirlines.Flight_trans_12.var() # variance
EastWestAirlines.Flight_trans_12.std() #standard deviation
range = max(EastWestAirlines.Flight_trans_12) - min(EastWestAirlines.Flight_trans_12) # range
range

#Days_since_enroll
EastWestAirlines.Days_since_enroll.var() # variance
EastWestAirlines.Days_since_enroll.std() #standard deviation
range = max(EastWestAirlines.Days_since_enroll) - min(EastWestAirlines.Days_since_enroll) # range
range

# Third moment business decision
EastWestAirlines.Balance.skew() # +ve skew , right skew
EastWestAirlines.Qual_miles.skew() # +ve skew , right skew
EastWestAirlines.cc1_miles.skew() # +ve skew , right skew
EastWestAirlines.Bonus_miles.skew() # +ve skew , right skew
EastWestAirlines.Bonus_trans.skew() # +ve skew , right skew
EastWestAirlines.Flight_miles_12mo.skew() # +ve skew , right skew
EastWestAirlines.Flight_trans_12.skew() # +ve skew , right skew
EastWestAirlines.Days_since_enroll.skew() # +ve skew , right skew


#Fourth moment business decision
EastWestAirlines.Balance.kurt() # Leptokurtic 
EastWestAirlines.Qual_miles.kurt() #Leptokurtic 
EastWestAirlines.cc1_miles.kurt() # platykurtic
EastWestAirlines.Bonus_miles.kurt() #Leptokurtic
EastWestAirlines.Bonus_trans.kurt() #Leptokurtic
EastWestAirlines.Flight_miles_12mo.kurt() #Leptokurtic
EastWestAirlines.Flight_trans_12.kurt() #Leptokurtic
EastWestAirlines.Days_since_enroll.kurt() # platykurtic

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np


plt.hist(EastWestAirlines.Balance) #histogram
plt.hist(EastWestAirlines.Qual_miles)
plt.hist(EastWestAirlines.cc1_miles)
plt.hist(EastWestAirlines.Bonus_miles)
plt.hist(EastWestAirlines.Bonus_trans)
plt.hist(EastWestAirlines.Flight_miles_12mo)
plt.hist(EastWestAirlines.Flight_trans_12)
plt.hist(EastWestAirlines.Days_since_enroll)

#boxplot
plt.boxplot(EastWestAirlines.Balance,1,vert=False) #it has outliers and it is right skew
# mean is > median - right skew , difference of mean and median is very high outliers exist
# mean=73601.32758189547 , median= 43097.0
plt.boxplot(EastWestAirlines.Qual_miles,1,vert=False) # it has outliers and it is right skew
# mean is > median - right skew , difference of mean and median is very high outliers exist
# mean = 144.11452863215803, median = 0.0
plt.boxplot(EastWestAirlines.cc1_miles,1,vert=False) # no outliers and it is right skew
# mean is not far from median so there is no ouliers exist
#mean > median - right skew
#mean = 2.0595148787196798, median = 1.0
plt.boxplot(EastWestAirlines.Bonus_miles,1,vert=False) # it has outliers and it is right skew
# mean is approx same to the median because mean is not that much affected by ouliers (ouliers may be less)
plt.boxplot(EastWestAirlines.Bonus_trans,1,vert=False)# it has outliers and it is right skew
# mean should be greater then median but not,because mean is not that much affected by outliers (outliers may less)
# mean = 11.60190047511878, median = 12.0
plt.boxplot(EastWestAirlines.Flight_miles_12mo,1,vert=False) # it has outliers and it is right skew
# mean is > median - right skew , difference of mean and median is very high outliers exist
# mean = 460.05576394098523, median = 0.0
plt.boxplot(EastWestAirlines.Flight_trans_12,1,vert=False) # it has outliers and it is right skew
# mean is > median -  right skew
# mean = 1.3735933983495874, median = 0.0
 plt.boxplot(EastWestAirlines.Days_since_enroll,1,vert=False) # no outliers

#Normal Quantile-Quantile Plot

import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
#Balance
stats.probplot(EastWestAirlines.Balance, dist='norm',plot=pylab) #pylab is visual representation
#transformation to make workex variable normal
import numpy as np
stats.probplot(np.log(EastWestAirlines.Balance),dist="norm",plot=pylab) #best transformation
#Qual_miles
stats.probplot(EastWestAirlines.Qual_miles, dist='norm',plot=pylab)
stats.probplot(np.log(EastWestAirlines.Qual_miles),dist="norm",plot=pylab)
#Bonus_miles
stats.probplot(EastWestAirlines.Bonus_miles, dist='norm',plot=pylab)
stats.probplot(np.log(EastWestAirlines.Bonus_miles),dist="norm",plot=pylab)
#Bonus_trans
stats.probplot(EastWestAirlines.Bonus_trans, dist='norm',plot=pylab)
stats.probplot(np.log(EastWestAirlines.Bonus_trans),dist="norm",plot=pylab)
#Flight_miles_12mo
stats.probplot(EastWestAirlines.Flight_miles_12mo, dist='norm',plot=pylab)
stats.probplot(np.log(EastWestAirlines.Flight_miles_12mo),dist="norm",plot=pylab)
#Days_since_enroll
stats.probplot(EastWestAirlines.Days_since_enroll, dist='norm',plot=pylab)
stats.probplot(np.sqrt(EastWestAirlines.Days_since_enroll),dist="norm",plot=pylab)

###### Data Preprocessing########################################

## import packages
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

EastWestAirlines.isna().sum()
EastWestAirlines.describe() # f na values then count will decrease
EastWestAirlines.info() #data types , #object - categorical data

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min()) # or denominator (i.max()-i.min())
    return(x)
# if we have more 0's and 1's in the data we go to Normalization
 
# Normalized data frame (considering the numerical part of data)

EastWestAirlines1 = EastWestAirlines.iloc[:,[1,2,6,7,8,9,10]]
EastWestAirlines1.head()

EastWestAirlines_norm = norm_func(EastWestAirlines1)
EastWestAirlines_norm.describe() # min=0, max=1

###################### Outlier Treatment #########
"Balance, Qual_miles, Bonus_miles, Bonus_trans, Flight_miles_12mo, Flight_trans_12"
# so we have 6 variables which has outliers
# we leave some binary variables because that is categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)
EastWestAirlines1.dtypes
EastWestAirlines1.isna().sum()

# let's find outliers 
"Balance"
sns.boxplot(EastWestAirlines1.Balance);plt.title('Boxplot');plt.show()

###################### Winsorization #####################################
import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Balance'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
EastWestAirlines1_t = winsorizer.fit_transform(EastWestAirlines1[['Balance']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(EastWestAirlines1_t.Balance);plt.title('Balance');plt.show()

#we see no outiers

"Qual_miles"
#let's find outliers
sns.boxplot(EastWestAirlines1.Qual_miles);plt.title('Boxplot');plt.show()

###################### Winsorization #####################################
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Qual_miles'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
EastWestAirlines1_t = winsorizer.fit_transform(EastWestAirlines1[['Qual_miles']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(EastWestAirlines1_t.Qual_miles);plt.title('Boxplot');plt.show()
#we see no outiers

"Bonus_miles"
#let's find outliers
sns.boxplot(EastWestAirlines1.Bonus_miles);plt.title('Boxplot');plt.show()
# Detection of Outliers
IQR = EastWestAirlines1['Bonus_miles'].quantile(0.75) - EastWestAirlines1['Bonus_miles'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = EastWestAirlines1['Bonus_miles'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = EastWestAirlines1['Bonus_miles'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

####################### Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
EastWestAirlines1['EastWestAirlines_replaced']= pd.DataFrame(np.where(EastWestAirlines1['Bonus_miles'] > upper_limit, upper_limit,
                                         np.where(EastWestAirlines1['Bonus_miles'] < lower_limit, lower_limit,
                                                  EastWestAirlines1['Bonus_miles'])))
                                 
sns.boxplot(EastWestAirlines1.EastWestAirlines_replaced);plt.title('Boxplot');plt.show()

#we see no outliers

"Bonus_trans"
#let's find outliers
sns.boxplot(EastWestAirlines1.Bonus_trans);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Bonus_trans'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
EastWestAirlines1_t = winsorizer.fit_transform(EastWestAirlines1[['Bonus_trans']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(EastWestAirlines1_t.Bonus_trans);plt.title('Boxplot');plt.show()
#we see no outiers

"Flight_miles_12mo"

#let's find outliers
sns.boxplot(EastWestAirlines1.Flight_miles_12mo);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Flight_miles_12mo'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
EastWestAirlines1_t = winsorizer.fit_transform(EastWestAirlines1[['Flight_miles_12mo']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(EastWestAirlines1_t.Flight_miles_12mo);plt.title('Boxplot');plt.show()
#we see no outiers

"Flight_trans_12"

#let's find outliers
sns.boxplot(EastWestAirlines1.Flight_trans_12);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Flight_trans_12'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
EastWestAirlines1_t = winsorizer.fit_transform(EastWestAirlines1[['Flight_trans_12']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(EastWestAirlines1_t.Flight_trans_12);plt.title('Boxplot');plt.show()
#we see no outiers

#################### Missing Values Imputation ##################################
# check for count of NA'sin each column
EastWestAirlines1.isna().sum()

# there is no na values

################## Type casting###############################################
#Identify duplicates records in the data
duplicate = EastWestAirlines1.duplicated()
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
    kmeans.fit(EastWestAirlines_norm)
    TWSS.append(kmeans.inertia_) # total within sum of square (variance)
    
TWSS
# Scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(EastWestAirlines_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
EastWestAirlines1['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
EastWestAirlines2 = EastWestAirlines1.iloc[:,[7,0,1,2,3,4,5,6]]
EastWestAirlines2.head()
EastWestAirlines_norm.head()

EastWestAirlines2.iloc[:, 1:].groupby(EastWestAirlines1.clust).mean()

EastWestAirlines2.to_csv("EastWestAirlinesoutput1.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/K Means Clustering"
os.chdir(path) # current working directory

"We are trying to learn more about EastWest Airlines' customers                 
"based on their flying patterns, earning and use of frequent flyer     
"rewards, and use of the airline credit card.
"The primary objective is to identify customer segments via KMeans clustering       
"and design targeted marketing campaigns for each segment."

"In this K-Means clustering:-
"customer segment1 : 0th group (Highest number)
"customer segment2 : 2nd group (second Highest number)
"customer segment3 : 1st group (third Highest number)

# now k = 4

model = KMeans(n_clusters = 4)
model.fit(EastWestAirlines_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
EastWestAirlines1['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
EastWestAirlines3 = EastWestAirlines1.iloc[:,[7,0,1,2,3,4,5,6]]
EastWestAirlines3.head()
EastWestAirlines_norm.head()

EastWestAirlines3.iloc[:, 1:].groupby(EastWestAirlines3.clust).mean()

EastWestAirlines3.to_csv("EastWestAirlinesoutput2.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/K Means Clustering"
os.chdir(path) # current working directory

"In this K-Means clustering:-
"customer segment1 : 3rd group (Highest number)
"customer segment2 : 0th group (second Highest number)
"customer segment3 : 2nd group (third Highest number)
"customer segment4 : 1st group (fourth Highest number)

"6. Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided. "

"On the basis KMeans clustering we know more about the EastWest Airlines' customers                 
"based on their flying patterns, earning and use of frequent flyer     
"rewards, and use of the airline credit card. we know more about them by 
"identifying customer segments via KMeans clustering       
"and design targeted marketing campaigns for each segment.
"Now, we can more figure out which segment has more profit and which has less"