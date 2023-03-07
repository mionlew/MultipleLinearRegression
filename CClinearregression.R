#Predicting credit card rating based on variables through multiple logistic regression


#Load libraries and read in csv files
library(tidyverse)
library(olsrr)
library(lm.beta)
library(car)
credit <- read.csv("Downloads/Credit.csv")
ccr_prediction <- read.csv("Downloads/OneDrive - California State University, Sacramento/creditrating_prediction.csv")

#Sum of missing data
sum(is.na(credit)) #No missing data





#Explore the dataset
View(credit)

#Summary stats for entire dataset
summary(credit)

#Scatter plot of relationship between age and credit rating.
ggplot(data = credit, mapping = aes(x = Age, y = Rating)) +
  geom_point() +
  geom_smooth(method = 'lm') #No correlation shown visually

#Correlation coefficient relationship for age and rating
cor(credit$Age, credit$Rating) #Low correlation rate indicating no real correlation

#Scatter plot of relationship between education level and credit rating
ggplot(data = credit, mapping = aes(x = Education, y = Rating)) +
  geom_point() +
  geom_smooth(method = 'lm')

#Correlation coefficient relationship for education and rating
cor(credit$Education, credit$Rating) #No correlation





#Multiple linear regression analysis
#Partition the dataset into a training and validation set 50/50
set.seed(42)
samplesplit <- sample(c(TRUE, FALSE), nrow(credit), replace = TRUE, prob = c(0.50, 0.50))
trainingset <- credit[samplesplit, ]
validationset <- credit[!samplesplit, ]

#Correlation matrix of all quantitative variables in the training set
cor(trainingset) #Coefficients of 0.82 and 0.99 show strong correlation of income and limit to rating

#Outcome variable as rating to perform multiple linear regression
rating_mlr <- lm(Rating ~ Income + Limit + Balance + Age + Education + Student + Gender + Married, data = trainingset)
#View the linear regression's statistic summary
summary(rating_mlr)
#View the variance inflation factor to determine if multicollinearity is appearing
vif(rating_mlr) #High correlation between income, limit, and balance


#Perform another multiple linear regression analysis taking out one of the high rating vifs
rating1_mlr <- lm(Rating ~ Income + Age + Education + Student + Gender + Married, data = trainingset)
summary(rating1_mlr) #Removing the multicollinearity (removing variable influence) increases correlation with other factors

#Vector of predicted values and residuals
rating_pred <- predict(rating1_mlr)
rating_res <- resid(rating1_mlr)
#Dataframe of predicted and residual values
rating_predres <- data.frame(rating_pred, rating_res)
#Scatter plot of predicted and residual values
ggplot(data = rating_predres, mapping = aes(x = rating_pred, y = rating_res)) +
  geom_point() +
  labs(title = "Residual vs. Predicted Values", x = "Predictions", y = "Residuals")

#Standardized residuals for a qq normal probability plot
rating_standard.res <- rstandard(rating1_mlr)
#Normal scores 
qqnorm(rating_standard.res, ylab = "Standardized Residuals", xlab = "Normal Scores")
#The qq plot looks good, should be a diagonal line across the plot, which shows normally distributed residuals

#Check for variables that are statistically significant
summary(rating_mlr) #Income, balance, age, and student are significant, indicated by the p scores




#Multiple linear regression using income, balance, age, and student
rating2_mlr <- lm(Rating ~ Income + Balance + Age + Student, data = trainingset)
summary(rating2_mlr) #Adjusted R squared means the variables included in this regression account for 99% of the variability in rating
#The correlation coefficients tell a story; for every one unit increase in balance, holding all other variables constant, predicted rating will increase 20.22 points
#Standardized
lm.beta(rating2_mlr) #Highest correlation when standardized is balance followed by income


#Final regression analysis using the validation set
validation_mlr <- lm(Rating ~ Income + Balance + Age + Student, data = validationset)
summary(validation_mlr)
#95% default prediction interval to predict ratings for a new dataset of individuals
predict(validation_mlr, ccr_prediction, interval = "prediction", level = 0.95)
