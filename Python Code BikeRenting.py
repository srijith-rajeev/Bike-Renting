#Load libraries
import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

#Set working directory
os.chdir("D:/Data Scientist/Edwisor/Project/Project 1")
df = pd.read_csv("day.csv")

#Getting the number of variables and obervation in the datasets
df.shape

#Getting the column names of the dataset
df.columns

#Getting the describtion of the dataset
df.describe()

#Getting the structure of the dataset
df.info()

#Getting first 5 rows of the dataset
df.head(5)

#missing value analysis
missing_val = pd.DataFrame(df.isnull().sum())
missing_val
# no missing values
#Removing redundant variables
#Since working day is defined in terms of weekday and holiday, the variable workingday is redundant. 
df.iloc[:,5:8].drop_duplicates()
#So removing the variable workingday
df = df.drop('workingday', 1)

# instant and dteday carries same information for our analysis.
pd.to_datetime(df['dteday'], format = "%Y-%m-%d")

#so removing dteday
df = df.drop('dteday',1)

# Encoding categorical data
df.iloc[:,1:7] = df.iloc[:,1:7].astype('category')
df.info()

# no scaling required for this data

#checking normality of numeric data


#oulier analysis
# ## BoxPlots - Distribution and Outlier Check
%matplotlib inline  
import matplotlib.pyplot as plt
plt.boxplot(df['temp'])
plt.title("Outlier Analysis of Temp")

plt.boxplot(df['atemp'])
plt.title("Outlier Analysis of ATemp")

plt.boxplot(df['hum'])
plt.title("Outlier Analysis of Hum")
#hum has oulier

plt.boxplot(df['windspeed'])
plt.title("Outlier Analysis of Windspeed")
#windspeed has oulier

#Remove outliers using boxplot method
#finding number of outliers
# outliers exist for hum, windspeed

#replacing outliers of hum with na
#Extract quartiles
q75, q25 = np.percentile(df['hum'], [75 ,25])
#Calculate IQR
iqr = q75 - q25
#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)
#Replace with NA
df.loc[df['hum'] < minimum, ['hum']] = np.nan
df.loc[df['hum'] > maximum, ['hum']] = np.nan
#Calculate missing value
missing_val = pd.DataFrame(df.isnull().sum())
missing_val

#replacing outliers of windspeed with na
#Extract quartiles
q75, q25 = np.percentile(df['windspeed'], [75 ,25])
#Calculate IQR
iqr = q75 - q25
#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)
#Replace with NA
df.loc[df['windspeed'] < minimum, ['windspeed']] = np.nan
df.loc[df['windspeed'] > maximum, ['windspeed']] = np.nan
#Calculate missing value
missing_val = pd.DataFrame(df.isnull().sum())
missing_val


#Impute with KNN
#from fancyimpute import KNN  
# fancyimpute not getting installed 

from fancyimpute import KNN
df = pd.DataFrame(KNN(k = 3).complete(df), columns = df.columns)

#as fancyimpute not getting installed - usin imputed file output of R
df.to_csv("bikerenting_Outlier.csv", index = False)
df = pd.read_csv("bikerenting_withoutOutlier.csv")

missing_val = pd.DataFrame(df.isnull().sum())
missing_val
# preparing indicator variables
#giving names to codes in categorical variables
df['season'] = df['season'].replace(1, 'Springer')
df['season'] = df['season'].replace(2, 'Summer')
df['season'] = df['season'].replace(3, 'Fall')
df['season'] = df['season'].replace(4, 'Winter')

df['mnth'] = df['mnth'].replace(1, 'Jan')
df['mnth'] = df['mnth'].replace(2, 'Feb')
df['mnth'] = df['mnth'].replace(3, 'Mar')
df['mnth'] = df['mnth'].replace(4, 'Apr')
df['mnth'] = df['mnth'].replace(5, 'May')
df['mnth'] = df['mnth'].replace(6, 'Jun')
df['mnth'] = df['mnth'].replace(7, 'Jul')
df['mnth'] = df['mnth'].replace(8, 'Aug')
df['mnth'] = df['mnth'].replace(9, 'Sep')
df['mnth'] = df['mnth'].replace(10, 'Oct')
df['mnth'] = df['mnth'].replace(11, 'Nov')
df['mnth'] = df['mnth'].replace(12, 'Dec')

df['weekday'] = df['weekday'].replace(0, 'Sun')
df['weekday'] = df['weekday'].replace(1, 'Mon')
df['weekday'] = df['weekday'].replace(2, 'Tue')
df['weekday'] = df['weekday'].replace(3, 'Wed')
df['weekday'] = df['weekday'].replace(4, 'Thu')
df['weekday'] = df['weekday'].replace(5, 'Fri')
df['weekday'] = df['weekday'].replace(6, 'Sat')

df['weathersit'] = df['weathersit'].replace(1, 'Clear')
df['weathersit'] = df['weathersit'].replace(2, 'Mist')
df['weathersit'] = df['weathersit'].replace(3, 'Light')
df['weathersit'] = df['weathersit'].replace(4, 'Heavy')


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

lb_results_df = pd.DataFrame(lb.fit_transform(df['season']), columns=lb.classes_)
df = df.drop('season',1)
df = pd.concat([df, lb_results_df[['Fall','Summer','Winter']]], axis=1)

lb_results_df = pd.DataFrame(lb.fit_transform(df['mnth']), columns=lb.classes_)
df = df.drop('mnth',1)
df = pd.concat([df, lb_results_df[['Feb','Mar', 'Apr','May', 'Jun','Jul', 'Aug','Sep', 'Oct','Nov', 'Dec']]], axis=1)

lb_results_df = pd.DataFrame(lb.fit_transform(df['weekday']), columns=lb.classes_)
df = df.drop('weekday',1)
df = pd.concat([df, lb_results_df[['Mon','Tue','Wed','Thu','Fri','Sat']]], axis=1)

lb_results_df = pd.DataFrame(lb.fit_transform(df['weathersit']), columns=lb.classes_)
df = df.drop('weathersit',1)
df = pd.concat([df, lb_results_df[['Mist','Light']]], axis=1)



#Normality check
df.hist(column = 'temp')
df.hist(column = 'atemp')
df.hist(column = 'hum')
df.hist(column = 'windspeed')
df.hist(column = 'casual')
df.hist(column = 'registered')
df.hist(column = 'cnt')

#histogram shows that distribution of casual and registered is different. So seperate analaysis is reqd
# also distribution of casual and registered is different for working days,weekdays, holidays


# coorelation between numeric variables
df_numeric = pd.DataFrame(df, columns=['instant','temp','atemp','hum','windspeed'])
plt.matshow(df_numeric.corr())
df_numeric.corr()
#temp and atemp are highly correlated. So dropping temp
df=df.drop(labels = 'temp',axis = 1)
df.info()


# anova and further variable selection will be done after fitting the model


#Simple Random Sampling
train,test = train_test_split(df,test_size = 0.2, random_state = 0)


# will check errors of the model probably by MAPE and MSE
#MAPE
#function calculate MAPE
def MAPE(y, yhat):
    mape = np.mean(np.abs((y - yhat)*100/y))
    return mape
#function MSE
def MSE(y, yhat):
    mse = np.mean(((y - yhat)**2)/len(yhat))
    return mse  

#adding intercept column for linear regression
train.insert(loc=0, column='intercept', value=1)
test.insert(loc=0, column='intercept', value=1)
independentVar = ['intercept','instant','yr','holiday','atemp','hum','windspeed','Fall','Summer','Winter','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Mon','Tue','Wed','Thu','Fri','Sat','Mist','Light']
train_independentVar = pd.DataFrame(train, columns= independentVar)
test_independentVar = pd.DataFrame(test, columns= independentVar)
import statsmodels.api as sm

#Linear Regression for Casual
regressor_casual = sm.OLS(train.loc[:,'casual'], train_independentVar).fit()
regressor_casual.summary()
#all variables are significant
# Predicting the Test set results
y_casual_linreg = regressor_casual.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'casual'], y_casual_linreg)
#Error: 90.82 %
#MSE
MSE(test.loc[:,'casual'], y_casual_linreg)
#MSE 959

#Linear Regression for registered
regressor_registered = sm.OLS(train.loc[:,'registered'], train_independentVar).fit()
regressor_registered.summary()
#all variables are significant
# Predicting the Test set results
y_registered_linreg = regressor_registered.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'registered'], y_registered_linreg)
#Error: 17.31 %
#MSE
MSE(test.loc[:,'registered'], y_registered_linreg)
#MSE 2475

#to get cnt , casual and registered is added
y_combined_linreg = y_casual_linreg +y_registered_linreg
#MAPE
MAPE(test.loc[:,'cnt'], y_combined_linreg)
#Error: 17.38 %
#MSE
MSE(test.loc[:,'cnt'], y_combined_linreg)
#MSE 4051

#Linear Regression for cnt
regressor_cnt = sm.OLS(train.loc[:,'cnt'], train_independentVar).fit()
regressor_cnt.summary()
#all variables are significant
# Predicting the Test set results
y_cnt_linreg = regressor_cnt.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'cnt'], y_cnt_linreg)
#Error: 17.38 %
#MSE
MSE(test.loc[:,'cnt'], y_cnt_linreg)
#MSE 4051

from sklearn.tree import DecisionTreeRegressor
#intercept is not needed in dataset for decision tree and random forest
independentVar = ['instant','yr','holiday','atemp','hum','windspeed','Fall','Summer','Winter','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Mon','Tue','Wed','Thu','Fri','Sat','Mist','Light']
# decision tree for casual
dectre_casual = DecisionTreeRegressor(max_depth=2,random_state=0).fit(train_independentVar, train.loc[:,'casual'])
y_casual_dectre = dectre_casual.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'casual'], y_casual_dectre)
#Error: 101.244 %
#MSE
MSE(test.loc[:,'casual'], y_casual_dectre)
#MSE 1858.04



# decision tree for registered
dectre_registered = DecisionTreeRegressor(max_depth=2,random_state=0).fit(train_independentVar, train.loc[:,'registered'])
y_registered_dectre = dectre_registered.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'registered'], y_registered_dectre)
#Error: 37.12 %
#MSE
MSE(test.loc[:,'registered'], y_registered_dectre)
#MSE 7595.55


#to get cnt , casual and registered is added
y_combined_dectre = y_casual_dectre +y_registered_dectre
#MAPE
MAPE(test.loc[:,'cnt'], y_combined_dectre)
#Error: 32.97 %
#MSE
MSE(test.loc[:,'cnt'], y_combined_dectre)
#MSE 8490.96

# decision tree for cnt
dectre_cnt = DecisionTreeRegressor(max_depth=2,random_state=0).fit(train_independentVar, train.loc[:,'cnt'])
y_cnt_dectre = dectre_cnt.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'cnt'], y_cnt_dectre)
#Error: 35.28 %
#MSE
MSE(test.loc[:,'cnt'], y_cnt_dectre)
#MSE 9365.23


#random forest
from sklearn.ensemble import RandomForestRegressor
randfor = RandomForestRegressor(n_estimators = 10, random_state = 0)
# random forest for casual
randfor.fit(train_independentVar,train.loc[:,'casual'])
y_casual_randfor = randfor.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'casual'], y_casual_randfor)
#Error: 61.9 %
#MSE
MSE(test.loc[:,'casual'], y_casual_randfor)
#MSE 1152

# random forest for registered
randfor.fit(train_independentVar,train.loc[:,'registered'])
y_registered_randfor = randfor.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'registered'], y_registered_randfor)
#Error: 22 %
#MSE
MSE(test.loc[:,'registered'], y_registered_randfor)
#MSE 3434

#to get cnt , casual and registered is added
y_combined_randfor = y_casual_randfor +y_registered_randfor
#MAPE
MAPE(test.loc[:,'cnt'], y_combined_randfor)
#Error: 18.91 %
#MSE
MSE(test.loc[:,'cnt'], y_combined_randfor)
#MSE 3432

# random forest for cnt
randfor.fit(train_independentVar,train.loc[:,'cnt'])
y_cnt_randfor = randfor.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'cnt'], y_cnt_randfor)
#Error: 18.75 %
#MSE
MSE(test.loc[:,'cnt'], y_cnt_randfor)
#MSE 2826

#for decision tree and random forest , instant doesn't make sense for predictiong future data.
# so fitting  model without instant

independentVar = ['intercept','yr','holiday','atemp','hum','windspeed','Fall','Summer','Winter','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Mon','Tue','Wed','Thu','Fri','Sat','Mist','Light']
train_independentVar = pd.DataFrame(train, columns= independentVar)
test_independentVar = pd.DataFrame(test, columns= independentVar)
import statsmodels.api as sm

#Linear Regression for Casual
regressor_casual = sm.OLS(train.loc[:,'casual'], train_independentVar).fit()
regressor_casual.summary()
#all variables are significant
# Predicting the Test set results
y_casual_linreg = regressor_casual.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'casual'], y_casual_linreg)
#Error: 90 %
#MSE
MSE(test.loc[:,'casual'], y_casual_linreg)
#MSE 958

#Linear Regression for registered
regressor_registered = sm.OLS(train.loc[:,'registered'], train_independentVar).fit()
regressor_registered.summary()
#all variables are significant
# Predicting the Test set results
y_registered_linreg = regressor_registered.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'registered'], y_registered_linreg)
#Error: 17.44 %
#MSE
MSE(test.loc[:,'registered'], y_registered_linreg)
#MSE 2489

#to get cnt , casual and registered is added
y_combined_linreg = y_casual_linreg +y_registered_linreg
#MAPE
MAPE(test.loc[:,'cnt'], y_combined_linreg)
#Error: 17 %
#MSE
MSE(test.loc[:,'cnt'], y_combined_linreg)
#MSE 4067

#Linear Regression for cnt
regressor_cnt = sm.OLS(train.loc[:,'cnt'], train_independentVar).fit()
regressor_cnt.summary()
#all variables are significant
# Predicting the Test set results
y_cnt_linreg = regressor_cnt.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'cnt'], y_cnt_linreg)
#Error: 17 %
#MSE
MSE(test.loc[:,'cnt'], y_cnt_linreg)
#MSE 4067

from sklearn.tree import DecisionTreeRegressor
independentVar = ['yr','holiday','atemp','hum','windspeed','Fall','Summer','Winter','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Mon','Tue','Wed','Thu','Fri','Sat','Mist','Light']

# decision tree for casual
dectre_casual = DecisionTreeRegressor(max_depth=2,random_state=0).fit(train_independentVar, train.loc[:,'casual'])
y_casual_dectre = dectre_casual.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'casual'], y_casual_dectre)
#Error: 101.244 %
#MSE
MSE(test.loc[:,'casual'], y_casual_dectre)
#MSE 1858.04



# decision tree for registered
dectre_registered = DecisionTreeRegressor(max_depth=2,random_state=0).fit(train_independentVar, train.loc[:,'registered'])
y_registered_dectre = dectre_registered.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'registered'], y_registered_dectre)
#Error: 38.40 %
#MSE
MSE(test.loc[:,'registered'], y_registered_dectre)
#MSE 6977.72


#to get cnt , casual and registered is added
y_combined_dectre = y_casual_dectre +y_registered_dectre
#MAPE
MAPE(test.loc[:,'cnt'], y_combined_dectre)
#Error: 34.49 %
#MSE
MSE(test.loc[:,'cnt'], y_combined_dectre)
#MSE 8343.146

# decision tree for cnt
dectre_cnt = DecisionTreeRegressor(max_depth=2,random_state=0).fit(train_independentVar, train.loc[:,'cnt'])
y_cnt_dectre = dectre_cnt.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'cnt'], y_cnt_dectre)
#Error: 36.11 %
#MSE
MSE(test.loc[:,'cnt'], y_cnt_dectre)
#MSE 9499.145


#random forest
from sklearn.ensemble import RandomForestRegressor
randfor = RandomForestRegressor(n_estimators = 10, random_state = 0)
# random forest for casual
randfor.fit(train_independentVar,train.loc[:,'casual'])
y_casual_randfor = randfor.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'casual'], y_casual_randfor)
#Error: 58 %
#MSE
MSE(test.loc[:,'casual'], y_casual_randfor)
#MSE 1034

# random forest for registered
randfor.fit(train_independentVar,train.loc[:,'registered'])
y_registered_randfor = randfor.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'registered'], y_registered_randfor)
#Error: 25.4 %
#MSE
MSE(test.loc[:,'registered'], y_registered_randfor)
#MSE 3941

#to get cnt , casual and registered is added
y_combined_randfor = y_casual_randfor +y_registered_randfor
#MAPE
MAPE(test.loc[:,'cnt'], y_combined_randfor)
#Error: 21 %
#MSE
MSE(test.loc[:,'cnt'], y_combined_randfor)
#MSE 4360.165

# random forest for cnt
randfor.fit(train_independentVar,train.loc[:,'cnt'])
y_cnt_randfor = randfor.predict(test_independentVar)
#MAPE
MAPE(test.loc[:,'cnt'], y_cnt_randfor)
#Error: 22.38 %
#MSE
MSE(test.loc[:,'cnt'], y_cnt_randfor)
#MSE 4382
