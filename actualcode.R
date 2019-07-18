## set working directory ###
setwd("E:/Boston Housing")

##Import the data ##

test.boston <- read.csv("test.csv", header = TRUE)
train.boston <- read.csv("train.csv", header = TRUE)

## check missing values ##

library(e1071)
library(Hmisc)
describe(train.boston)

## check for skewness, as per the boxplot it should ideally be near to 0, but in this case it is found that##
summary(train.boston)
boxplot(train.boston)
library(e1071)
skewness(train.boston$ID)
skewness(train.boston$crim)
skewness(train.boston$zn)
skewness(train.boston$indus)
skewness(train.boston$chas)
skewness(train.boston$nox)
skewness(train.boston$rm)
skewness(train.boston$age)
skewness(train.boston$dis)
skewness(train.boston$rad)
skewness(train.boston$tax)
skewness(train.boston$ptratio)
skewness(train.boston$black)
skewness(train.boston$lstat)
skewness(train.boston$medv)

ggplot(train.boston)

##now we check for co-relations##
##  here we create a correlation matrix file and found that there are lot of variables which are colinear to each other##
##the usual co-linear is stronger to -1 or +1 ##
boxplot(train.boston)
cor_matrix <- cor(train.boston)
write.csv(cor_matrix,"cormat.csv")

###check for formats of the data ###

summary(train.boston)
str(train.boston)

##here we find that chas is in integer, since its just a factor, we convert it to factor ##

train.boston$chas <- as.factor(train.boston$chas)
str(train.boston)

## we write the linear model ##

linear_model <- lm(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data = train.boston)
linear_model


summary(linear_model)
## in the above model it is found that the least value is 11k and max is 24k, also few variables have -ve impact like nox, dis,ptratio etc ##

##now we predict the value for mean square error##
train.boston$pred_lm <- predict(linear_model, train.boston)
train.boston$res <- (train.boston$medv - train.boston$pred_lm)
train.boston$ressq <- (train.boston$res*train.boston$res)
meansq <- mean(train.boston$ressq)
rmse <- sqrt(meansq)

##Here we can see that the root mean squeared value is 4.7 i.e 4000$ off the housing price, so which means the model is not perfect, so we go for another model which is a decision tree as we have a continuous variable##
## if y is catergorical or flag in nature we use confusin matrix##

##use decision tree##
library(party)
png(file = "decision_tree.png")
names(train.boston)
model_tree <- ctree(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data = train.boston)
plot(model_tree)
dev.off()

summary(model_tree)
model_tree

train.boston$pred_tree <- predict(model_tree, train.boston)
mse_tree <- sqrt(mean((train.boston$medv-train.boston$pred_tree)^2))


###USING RANDOM FOREST ####
library("randomForest")
model_rf <- randomForest(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data = train.boston)
train.boston$pred_rf <- predict(model_rf, train.boston)
mse_rf <- sqrt(mean((train.boston$medv-train.boston$pred_rf)^2))
print(mse_rf)

##using SVM ###

library('e1071')
model_svm <- svm(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data = train.boston)
train.boston$pred_svm <- predict(model_svm, train.boston)
mse_svm <- sqrt(mean((train.boston$medv-train.boston$pred_svm)^2))
print(mse_svm)
print(model_svm)

### performing on the test data ###

str(test.boston)

##convert the chas variable into factor as its a dummy variable ##
test.boston$chas <- as.factor(test.boston$chas)
str(test.boston)

## we apply all our models on test data ###
##since random forest we got the least root mean squared, we used this model##
test.boston$pred_rf <- predict(model_rf,test.boston)
write.csv(test.boston$pred_rf,"submissionresult.csv")
