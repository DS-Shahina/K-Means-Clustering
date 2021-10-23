"Problem 5"

import pandas as pd

# Read data into Python
Auto = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Dataset_Assignment Clustering/AutoInsurance.csv")
Auto.head()
Auto.shape
Auto.columns.values
Auto.dtypes
Auto.info()

#  Delete column which is not necessary - 2 columns
# Only 22 columns is left
#let's remove columns which are not neended - we remove binary and categorical columns because it creates problem while clustering
Auto.drop(['Customer','Effective To Date'], axis='columns', inplace=True)
Auto.head()
Auto.shape
Auto.columns.values
Auto.dtypes
Auto.info()

#Exploratory Data Analysis
# Check the descriptive statistics of numeric variables
Auto.describe()
#1st moment Business Decision # Measures of Central Tendency / First moment business decision
Auto.mean()
Auto.median()
Auto.mode()

#2nd moment business decision # Measures of Dispersion / Second moment business decision
Auto.var() 
Auto.std()

#3rd moment Business Decision
Auto.skew()

#4th moment Business Decision
Auto.kurtosis()

#Dummy Vraibale Creation
#One Hot Encoding
# Creating dummy variables for categorical datatypes
Auto = pd.get_dummies(Auto, columns=['State', 'Response', 'EmploymentStatus', 'Gender','Location Code','Marital Status','Policy Type','Policy','Sales Channel','Vehicle Class'])

from sklearn.preprocessing import LabelEncoder
# Label Encoding -  'Coverage','Education','Renew Offer Type','Vehicle Size'
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
Auto = MultiColumnLabelEncoder(columns = ['Coverage','Education','Renew Offer Type','Vehicle Size']).fit_transform(Auto)

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np
import seaborn as sns
#Histogram
for i, predictor in enumerate(Auto):
    plt.figure(i)
    sns.histplot(data=Auto, x=predictor)
    
#boxplot    
for i, predictor in enumerate(Auto):
    plt.figure(i)
    sns.boxplot(data=Auto, x=predictor)

# Normalization function using z std. all are continuous data.
Auto1 = Auto[['Customer Lifetime Value','Income','Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Number of Open Complaints','Number of Policies','Total Claim Amount','Education','Vehicle Size']] 
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)
# Normalized data frame (considering the numerical part of data)
Auto1_norm = norm_func(Auto1) # we take numeric columns, becuase that binary varibales create problem while clustering
Auto1_norm.describe() # min=0, max=1
Auto1_norm.info()

###################### Outlier Treatment #########
# only for numerical data - which has outliers
'Customer Lifetime Value','Monthly Premium Auto','Number of Open Complaints','Number of Policies','Total Claim Amount'
# so we have 10 variables which has outliers
# we leave some binary variable because that is categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)

# let's find outliers 
"Customer Lifetime Value"
sns.boxplot(Auto1['Customer Lifetime Value']);plt.title('Boxplot');plt.show()

###################### Winsorization #####################################
import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Customer Lifetime Value'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Auto1_t = winsorizer.fit_transform(Auto1[['Customer Lifetime Value']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Auto1_t['Customer Lifetime Value']);plt.title('Customer Lifetime Value');plt.show()

#we see no outiers

"Monthly Premium Auto"
sns.boxplot(Auto1['Monthly Premium Auto']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Monthly Premium Auto'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Auto1_t = winsorizer.fit_transform(Auto1[['Monthly Premium Auto']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Auto1_t['Monthly Premium Auto']);plt.title('Monthly Premium Auto');plt.show()

#we see no outiers

"Number of Open Complaints"
sns.boxplot(Auto1['Number of Open Complaints']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Number of Open Complaints'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Auto1_t = winsorizer.fit_transform(Auto1[['Number of Open Complaints']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Auto1_t['Number of Open Complaints']);plt.title('Number of Open Complaints');plt.show()

#we see no outiers

"Number of Policies"
sns.boxplot(Auto1['Number of Policies']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Number of Policies'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Auto1_t = winsorizer.fit_transform(Auto1[['Number of Policies']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Auto1_t['Number of Policies']);plt.title('Number of Policies');plt.show()

#we see no outiers

"Total Claim Amount"
sns.boxplot(Auto1['Total Claim Amount']);plt.title('Boxplot');plt.show()

winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['Total Claim Amount'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
Auto1_t = winsorizer.fit_transform(Auto1[['Total Claim Amount']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

# lets see boxplot
sns.boxplot(Auto1_t['Total Claim Amount']);plt.title('Total Claim Amount');plt.show()

#we see no outiers

#################### Missing Values Imputation ##################################
# check for count of NA'sin each column
Auto1.isna().sum()

# there is no na values

################## Type casting###############################################
#Identify duplicates records in the data
duplicate = Auto1.duplicated()
sum(duplicate)

#Removing Duplicates
Auto2 = Auto1.drop_duplicates()

#there is no duplicate values in the data

# Model Building 
###### scree plot or elbow curve ############
from sklearn.cluster import KMeans
TWSS = []
k = list(range(2, 9)) #range is random
k

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(Auto1_norm)
    TWSS.append(kmeans.inertia_) # total within sum of square (variance)
    
TWSS
# Scree plot
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(Auto1_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
Auto1['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
Auto2 = Auto1.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]
Auto2.head()
Auto1_norm.head()

Auto2.iloc[:,1:].groupby(Auto2.clust).mean()

Auto2.to_csv("AutoInsurance_Dataoutput1.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/K Means Clustering"
os.chdir(path) # current working directory

"We are trying to learn more about Auto insurance pattern on the basis of various features,
"Basically we are trying to find out the auto insurance pattern using K-Means clustering. 
"The primary objective is to identify Auto insurance pattern segments via K-Means clustering and design targeted marketing campaigns for each segment."

"In this K-Means clustering:-
"Autoinsurance pattern1 : 2nd group (Highest number)
"Autoinsurance pattern2 : 0th group (second Highest number)
"Autoinsurance pattern3 : 1st group (third Highest number)

# now k = 4
# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(Auto1_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object  # series - column
Auto1['clust'] = mb # creating a  new column and assigning it to new column 
#mb- membership
Auto3 = Auto1.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]
Auto3.head()
Auto1_norm.head()

Auto3.iloc[:].groupby(Auto3.clust).mean()

Auto3.to_csv("AutoInsurance_Dataoutput2.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/K Means Clustering"
os.chdir(path) # current working directory

"In this K-Means clustering:-
"Autoinsurance pattern1 : 0th group (Highest number)
"Autoinsurance pattern2 : 2nd group (second Highest number)
"Autoinsurance pattern3 : 3rd group (third Highest number)
"Autoinsurance pattern4 : 1st group (fourth Highest number)

"6. Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided. "

"On the basis K-Means clustering We know more about Autoinsurance pattern based on some varibales, 
"Basically we are trying to find out the Autoinsurance pattern using K-Means clustering. 
"The primary objective is to identify Autoinsurance pattern segments via K-Means clustering and design targeted marketing campaigns for each segment.
"so, because of K-Means clustering we can eassily figure out which group has the highest Autoinsurance pattern and which has lowest, and we can also see individual variables and know about Autoinsurance pattern"

