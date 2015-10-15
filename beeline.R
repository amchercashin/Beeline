#library(readr)
library(caret)
library(h2o)
#library(mice)
# READ COMPETITION DATA:
{
train <- read.csv("./data/train.csv", na.strings = c("NA", ""))
test <- read.csv("./data/test.csv", na.strings = c("NA", ""))
}

train$x6 <- as.factor(train$x6); train$x7 <- as.factor(train$x7); train$x34 <- as.factor(train$x34)
test$x6 <- as.factor(test$x6); test$x7 <- as.factor(test$x7); test$x34 <- as.factor(test$x34)
train$y <- factor(train$y)

feature_names <- names(train)[1:(ncol(train)-1)]

train_no_na <- train[complete.cases(train),]

in_train <- sample(nrow(train_no_na), 20000)

## Start a local cluster with 2GB RAM
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '4g')
## 
train_h2o <- as.h2o(localH2O, train_no_na[in_train, ])
validation_h2o <- as.h2o(localH2O, train_no_na[-in_train, ])

system.time(
model <-        h2o.deeplearning(x = 1:62,  # column numbers for predictors
                                 y = 63,   # column number for label
                                 training_frame = train_h2o, # data in H2O format
                                 validation_frame = validation_h2o,
                                 #activation = "TanhWithDropout", # or 'Tanh'
                                 #input_dropout_ratio = 0.2, # % of inputs dropout
                                 #hidden_dropout_ratios = c(0.5,0.5,0.5), # % for nodes dropout
                                 #balance_classes = TRUE#, 
                                 #hidden = c(50,50,50), # three layers of 50 nodes
                                 epochs = 50
                                 ) # max. no. of epochs
)

## Using the DNN model for predictions
h2o_yhat_test <- h2o.predict(model, validation_h2o)

## Converting H2O format into data frame
df_yhat_test <- as.data.frame(h2o_yhat_test)


# system.time(X <- model.matrix(~., train_no_na[in_train, feature_names])[,-1])

# system.time(lasso_cv <- cv.glmnet(X, train_no_na$y[in_train], family = "gaussian", alpha = 1, 
#                                   type.measure = "auc", parallel = TRUE)) #27
# bestLambda <- lasso_cv$lambda.min
# bestLambdaCol <- which(lasso_cv$lambda==lasso_cv$lambda.min)
# lassoImpVars <- names(which(lasso_cv$glmnet.fit$beta[,bestLambdaCol]!=0))
# lassoImpVars <- ifelse(nchar(lassoImpVars) > 10, substr(lassoImpVars, 1, nchar(lassoImpVars)-10), lassoImpVars)
# lassoImpVars <- unique(lassoImpVars)

# new_x <- model.matrix(~., train_no_na[-in_train, feature_names])[,-1]
# new_y <- predict(lasso_cv, new_x, s="lambda.min", type = "response")
# new_y <- ifelse(new_y<1, 1, new_y)
# new_y <- round(new_y)
# new_y <- factor(new_y)
confusionMatrix(df_yhat_test[,1], train_no_na[-in_train, "y"])
