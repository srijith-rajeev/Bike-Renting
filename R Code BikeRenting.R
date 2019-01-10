#remove all the objects stored
rm(list=ls())

#set current working directory
setwd("D:/Data Scientist/Edwisor/Project/Project 1")

#Current working directory
getwd()

##Load data in R
#reading CSV
df = read.csv("day.csv", header = T) 

#Getting the number of variables and obervation in the datasets
dim(df)

#Getting the column names of the dataset
colnames(df)

#Getting the structure of the dataset
str(df)


#Getting first 5 rows of the dataset
head(df, 5)

#Unique values in a column
unique(df$instant)

#Count of unique values in a column
length(unique(df$mnth))

#Distribution of unique values in a column
table(df$mnth)

#Summary of data
summary(df)

#missing value analysis
missing_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
missing_val
# no missing values



#Removing redundant variables
#Since working day is defined in terms of weekday and holiday, the variable workingday is redundant. 
unique(df[,c(6:8)])
#So removing the variable workingday
df= df[-8]

# instant and dteday carries same information for our analysis.
df$dteday = as.numeric(as.Date(df$dteday))
sum((df$dteday - 14974 - df$instant)^2)
#so removing dteday
df = df[-2]

#Convert categorical variables as factors

df$season = factor(df$season, labels = c("Spring", "Summer", "Fall", "Winter"))
df$yr = factor(df$yr, labels = c("2011", "2012"))
df$mnth = factor(df$mnth, labels = c("Jan", "Feb","Mar", "Apr","May", "Jun",
                                     "Jul", "Aug","Sep", "Oct","Nov", "Dec"))
df$holiday = factor(df$holiday, labels = c("No", "Yes"))
df$weekday = factor(df$weekday, labels = c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"))
df$weathersit = factor(df$weathersit, levels = c(1,2,3,4), labels = c("Clear", "Mist", "Light", "Heavy"))
str(df)

summary(df)

# no scaling required for this data

#checking normality of numeric data


#oulier analysis
# ## BoxPlots - Distribution and Outlier Check
cnames_numeric = c("temp","atemp","hum","windspeed")

library(ggplot2)
library(gridExtra)
for (i in 1:length(cnames_numeric))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames_numeric[i])), data = subset(df))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames_numeric[i])+
           ggtitle(paste("Box plot of ",cnames_numeric[i])))
  
}
print(gn1)
print(gn2)
print(gn3)
#hum has oulier
print(gn4)
#windspeed has oulier



# # #Remove outliers using boxplot method
#finding number of outliers
# outliers exist for hum, windspeed
val =df$hum[df$hum %in% boxplot.stats(df$hum)$out]
cat('No of outliers is ' , length(val))
df$hum[df$hum %in% val] = NA
val =df$windspeed[df$windspeed %in% boxplot.stats(df$windspeed)$out]
cat('No of outliers is ' , length(val))
df$windspeed[df$windspeed %in% val] = NA

apply(df,2,function(x){sum(is.na(x))})

#knn imputation on missing values
#install.packages('DMwR')
library('DMwR')
df = knnImputation(df, k = 3)

#Normality check
qqnorm(df$hum)
hist(df$temp)
hist(df$atemp)
hist(df$hum)
hist(df$windspeed)
hist(df$casual)
hist(df$registered)
hist(df$cnt)
#histogram shows that distribution of casual and registered is different. So seperate analaysis is reqd
# casual and registered can be predicted seperately and then added to get cnt.
# also cnt can be directly predicted. 

#distribution of casual and registered is different for working days,weekdays, holidays

# coorelation between numeric variables
cor(df[c(1,8:11)]) 
#install.packages('corrplot')
library(corrplot)

corrplot(cor(df[c(1,8:11)]) , method="color")

#temp and atemp are highly correlated. So dropping temp
df = df[-8]


# anova and further variable selection will be done after fitting the model

#check multicollearity
#install.packages('usdm')
library(usdm)
vif(df[,c(1,8:10)])

vifcor(df[,c(1,8:10)], th = 0.8)


#Simple Random Sampling
set.seed(1234)
train_index = sample(1:nrow(df), 0.8 * nrow(df))
train = df[train_index,]
test = df[-train_index,]


# errors matrices  -  MAPE and MSE
#MAPE
#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)*100/y))
}
#MSE
MSE = function(y, yhat){
  mean(((y - yhat)^2)/length(y))
}


regressor_casual = lm(formula = casual ~ instant+season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = df)
summary(regressor_casual)
anova(regressor_casual)


#Linear Regression for Casual
regressor_casual = lm(formula = casual ~ instant+season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train)
summary(regressor_casual)
anova(regressor_casual)
#all variables are significant
# Predicting the Test set results
y_casual_linreg = predict(regressor_casual, newdata = test)
#all variables are significant
#MAPE
MAPE(test$casual, y_casual_linreg)
#Error: 148.67 %
#MSE
MSE(test$casual, y_casual_linreg)
#MSE 1095.517

#Linear Regression for Registered.
regressor_registered = lm(formula = registered ~ instant+season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train)
summary(regressor_registered)
anova(regressor_registered)
#all variables are significant
# Predicting the Test set results
y_registered_linreg = predict(regressor_registered, newdata = test)
#all variables are significant
#MAPE
MAPE(test$registered, y_registered_linreg)
#Error: 21.13 %
#MSE
MSE(test$registered, y_registered_linreg)
#MSE 2993.887

#to get cnt , casual and registered is added
y_combined_linreg = y_casual_linreg +y_registered_linreg
#MAPE
MAPE(test$cnt, y_combined_linreg)
#Error: 20.71 %
#MSE
MSE(test$cnt, y_combined_linreg)
#MSE 4644.176

#Linear Regression for Count.
regressor_cnt = lm(formula = cnt ~ instant+season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train)
summary(regressor_cnt)
anova(regressor_cnt)
#all variables are significant
# Predicting the Test set results
y_cnt_linreg = predict(regressor_cnt, newdata = test)
#all variables are significant
#MAPE
MAPE(test$cnt, y_cnt_linreg)
#Error: 20.71 %
#MSE
MSE(test$cnt, y_cnt_linreg)
#MSE 4644.176


library(rpart)

# decision tree for casual
dectre = rpart(casual ~ instant+season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train, method = "anova")
y_casual_dectre = predict(dectre, test[,1:10])
#MAPE
MAPE(test$casual, y_casual_dectre)
#Error: 81.91 %
#MSE
MSE(test$casual, y_casual_dectre)
#MSE 1326.286
plot(dectre)
text(dectre)

# decision tree for registered
dectre = rpart(registered ~ instant+season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train, method = "anova")
y_registered_dectre = predict(dectre, test[,1:10])
#MAPE
MAPE(test$registered, y_registered_dectre)
#Error: 23.14 %
#MSE
MSE(test$registered, y_registered_dectre)
#MSE 3965.553
plot(dectre)
text(dectre)

#to get cnt , casual and registered is added
y_combined_dectre = y_casual_dectre +y_registered_dectre
#MAPE
MAPE(test$cnt, y_combined_dectre)
#Error: 23.88 %
#MSE
MSE(test$cnt, y_combined_dectre)
#MSE 5954.94

# decision tree for cnt
dectre = rpart(cnt ~ instant+season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train, method = "anova")
y_cnt_dectre = predict(dectre, test[,1:10])
#MAPE
MAPE(test$cnt, y_cnt_dectre)
#Error: 25.05 %
#MSE
MSE(test$cnt, y_cnt_dectre)
#MSE 5924.238
#decision tree plotting
plot(dectre)
text(dectre)


#random forest
#install.packages('randomForest')
# random forest for casual
set.seed(1234)
randfor = randomForest::randomForest(x = train[1:10],
                       y = train$casual,
                       ntree = 200)
y_casual_randfor = predict(randfor, test[,1:10])
plot(randfor)
#MAPE
MAPE(test$casual, y_casual_randfor)
#Error: 57.94 %
#MSE
MSE(test$casual, y_casual_randfor)
#MSE 748.9912

# random forest for registered
randfor = randomForest::randomForest(x = train[1:10],
                                     y = train$registered,
                                     ntree = 200)
y_registered_randfor = predict(randfor, test[,1:10])
plot(randfor)
#MAPE
MAPE(test$registered, y_registered_randfor)
#Error: 17.35682 %
#MSE
MSE(test$registered, y_registered_randfor)
#MSE 2190.092

#to get cnt , casual and registered is added
y_combined_randfor = y_casual_randfor +y_registered_randfor
#MAPE
MAPE(test$cnt, y_combined_randfor)
#Error: 17.87429 %
#MSE
MSE(test$cnt, y_combined_randfor)
#MSE 3373.133

# random forest for cnt
randfor = randomForest::randomForest(x = train[1:10],
                                     y = train$cnt,
                                     ntree = 200)
y_cnt_randfor = predict(randfor, test[,1:10])
plot(randfor)
#MAPE
MAPE(test$cnt, y_cnt_randfor)
#Error: 18.38772 %
#MSE
MSE(test$cnt, y_cnt_randfor)
#MSE 3312.55


#for decision tree and random forest , instant doesn't make sense for predictiong future time seried data.
# so fitting  model without instant
train = train[-1]
test = test[-1]



#Linear Regression for Casual
regressor_casual = lm(formula = casual ~ season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train)
summary(regressor_casual)
anova(regressor_casual)
#all variables are significant
# Predicting the Test set results
y_casual_linreg = predict(regressor_casual, newdata = test)
#all variables are significant
#MAPE
MAPE(test$casual, y_casual_linreg)
#Error: 146.5538 %
#MSE
MSE(test$casual, y_casual_linreg)
#MSE 1085.127

#Linear Regression for Registered.
regressor_registered = lm(formula = registered ~ season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train)
summary(regressor_registered)
anova(regressor_registered)
#all variables are significant
# Predicting the Test set results
y_registered_linreg = predict(regressor_registered, newdata = test)
#all variables are significant
#MAPE
MAPE(test$registered, y_registered_linreg)
#Error: 20.8848 %
#MSE
MSE(test$registered, y_registered_linreg)
#MSE 3012.93

#to get cnt , casual and registered is added
y_combined_linreg = y_casual_linreg +y_registered_linreg
#MAPE
MAPE(test$cnt, y_combined_linreg)
#Error: 20.66 %
#MSE
MSE(test$cnt, y_combined_linreg)
#MSE 4640.17

#Linear Regression for Count.
regressor_cnt = lm(formula = cnt ~ season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train)
summary(regressor_cnt)
anova(regressor_cnt)
#all variables are significant
# Predicting the Test set results
y_cnt_linreg = predict(regressor_cnt, newdata = test)
#all variables are significant
#MAPE
MAPE(test$cnt, y_cnt_linreg)
#Error: 20.66 %
#MSE
MSE(test$cnt, y_cnt_linreg)
#MSE 4640.172


library(rpart)
# decision tree for casual
dectre = rpart(casual ~ season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train, method = "anova")
y_casual_dectre = predict(dectre, test[,1:10])
#MAPE
MAPE(test$casual, y_casual_dectre)
#Error: 81.86 %
#MSE
MSE(test$casual, y_casual_dectre)
#MSE 1327.462

# decision tree for registered
dectre = rpart(registered ~ season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train, method = "anova")
y_registered_dectre = predict(dectre, test[,1:10])
#MAPE
MAPE(test$registered, y_registered_dectre)
#Error: 26.82 %
#MSE
MSE(test$registered, y_registered_dectre)
#MSE 4396.738

#to get cnt , casual and registered is added
y_combined_dectre = y_casual_dectre +y_registered_dectre
#MAPE
MAPE(test$cnt, y_combined_dectre)
#Error: 26.61 %
#MSE
MSE(test$cnt, y_combined_dectre)
#MSE 6068.552

# decision tree for cnt
dectre = rpart(cnt ~ season+yr+mnth+holiday+weekday+weathersit+atemp+hum+windspeed , data = train, method = "anova")
y_cnt_dectre = predict(dectre, test[,1:10])
#MAPE
MAPE(test$cnt, y_cnt_dectre)
#Error: 22.81 %
#MSE
MSE(test$cnt, y_cnt_dectre)
#MSE 5407.002
#decision tree plotting
plot(dectre)
text(dectre)


#random forest
#install.packages('randomForest')
# random forest for casual
set.seed(1234)
randfor = randomForest::randomForest(x = train[1:10],
                                     y = train$casual,
                                     ntree = 200)
y_casual_randfor = predict(randfor, test[,1:10])
plot(randfor)
#MAPE
MAPE(test$casual, y_casual_randfor)
#Error: 22.92 %
#MSE
MSE(test$casual, y_casual_randfor)
#MSE 128.4586

# random forest for registered
randfor = randomForest::randomForest(x = train[1:10],
                                     y = train$registered,
                                     ntree = 200)
y_registered_randfor = predict(randfor, test[,1:10])
plot(randfor)
#MAPE
MAPE(test$registered, y_registered_randfor)
#Error: 17.107 %
#MSE
MSE(test$registered, y_registered_randfor)
#MSE 1815.706

#to get cnt , casual and registered is added
y_combined_randfor = y_casual_randfor +y_registered_randfor
#MAPE
MAPE(test$cnt, y_combined_randfor)
#Error: 15.41 %
#MSE
MSE(test$cnt, y_combined_randfor)
#MSE 2034.908

# random forest for cnt
randfor = randomForest::randomForest(x = train[1:10],
                                     y = train$cnt,
                                     ntree = 200)
y_cnt_randfor = predict(randfor, test[,1:10])
plot(randfor)
#MAPE
MAPE(test$cnt, y_cnt_randfor)
#Error: 17.1093 %
#MSE
MSE(test$cnt, y_cnt_randfor)
#MSE 2599.182

