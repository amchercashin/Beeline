#library(readr)
library(caret)
library(glmnet)
library(mice)
# READ COMPETITION DATA:
{
train <- read.csv("./data/train.csv", na.strings = c("NA", ""))
test <- read.csv("./data/test.csv", na.strings = c("NA", ""))
}

train$x6 <- as.factor(train$x6); train$x7 <- as.factor(train$x7); train$x34 <- as.factor(train$x34)
test$x6 <- as.factor(test$x6); test$x7 <- as.factor(test$x7); test$x34 <- as.factor(test$x34)
feature_names <- names(train)[1:(ncol(train)-1)]

train_no_na <- train[complete.cases(train),]

in_train <- sample(nrow(train_no_na), 20000)

system.time(X <- model.matrix(~., train_no_na[in_train, feature_names])[,-1])

system.time(lasso_cv <- cv.glmnet(X, train_no_na$y[in_train], family = "gaussian", alpha = 1, 
                                  type.measure = "auc", parallel = TRUE)) #27