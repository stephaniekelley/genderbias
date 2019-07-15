####ANTI-DISCRIMINATION LAWS AND ALGORITHMIC DECISION-MAKING####
#Authors: Stephanie Kelley and Anton Ovchinnikov
#Special thanks to our Research Assisant Harshdeep Singh for his work on Feature Engineering, and some of the more computationally complex models
#last updated July 15th, 2019

#ensure necessary packages are installed
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not
pacman::p_load("tidyverse", "ROCR", "caret", "xgboost", "magrittr") #Check, and if needed install the necessary packages

library(tidyverse)
library(ROCR)
library(caret)
library(xgboost)
library(magrittr)

#set a random number seed for use throughout analysis
set.seed(0)

####COMBINE DATASETS INTO "ORIGINAL DATA"####
#import Kaggle data from Home Credit Default Risk competition (saved from link below in workspace)
#datasets found at https://www.kaggle.com/c/home-credit-default-risk as of July 15, 2019


cat("Loading data...\n")

bbalance <- read_csv("../input/bureau_balance.csv") 
bureau <- read_csv("../input/bureau.csv")
cc_balance <- read_csv("../input/credit_card_balance.csv")
payments <- read_csv("../input/installments_payments.csv") 
pc_balance <- read_csv("../input/POS_CASH_balance.csv")
prev <- read_csv("../input/previous_application.csv")
tr <- read_csv("../input/application_train.csv") 
te <- read_csv("../input/application_test.csv")

cat("Preprocessing...\n")

fn <- funs(mean, sd, min, max, sum, n_distinct, .args = list(na.rm = TRUE))

sum_bbalance <- bbalance %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_BUREAU) %>% 
  summarise_all(fn) 

rm(bbalance); gc()

sum_bureau <- bureau %>% 
  left_join(sum_bbalance, by = "SK_ID_BUREAU") %>% 
  select(-SK_ID_BUREAU) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)

rm(bureau, sum_bbalance); gc()

sum_cc_balance <- cc_balance %>% 
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)
rm(cc_balance); gc()

sum_payments <- payments %>% 
  select(-SK_ID_PREV) %>% 
  mutate(PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT,
         PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT,
         DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
         DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT,
         DPD = ifelse(DPD > 0, DPD, 0),
         DBD = ifelse(DBD > 0, DBD, 0)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn) 
rm(payments); gc()

sum_pc_balance <- pc_balance %>% 
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)
rm(pc_balance); gc()

sum_prev <- prev %>%
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  mutate(DAYS_FIRST_DRAWING = ifelse(DAYS_FIRST_DRAWING == 365243, NA, DAYS_FIRST_DRAWING),
         DAYS_FIRST_DUE = ifelse(DAYS_FIRST_DUE == 365243, NA, DAYS_FIRST_DUE),
         DAYS_LAST_DUE_1ST_VERSION = ifelse(DAYS_LAST_DUE_1ST_VERSION == 365243, NA, DAYS_LAST_DUE_1ST_VERSION),
         DAYS_LAST_DUE = ifelse(DAYS_LAST_DUE == 365243, NA, DAYS_LAST_DUE),
         DAYS_TERMINATION = ifelse(DAYS_TERMINATION == 365243, NA, DAYS_TERMINATION),
         APP_CREDIT_PERC = AMT_APPLICATION / AMT_CREDIT) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn) 
rm(prev); gc()

tri <- 1:nrow(tr)
Target <- tr$TARGET

tr_te <- tr %>% 
  select(-TARGET) %>% 
  bind_rows(te) %>%
  left_join(sum_bureau, by = "SK_ID_CURR") %>% 
  left_join(sum_cc_balance, by = "SK_ID_CURR") %>% 
  left_join(sum_payments, by = "SK_ID_CURR") %>% 
  left_join(sum_pc_balance, by = "SK_ID_CURR") %>% 
  left_join(sum_prev, by = "SK_ID_CURR") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  mutate(na = apply(., 1, function(x) sum(is.na(x))),
         DAYS_EMPLOYED = ifelse(DAYS_EMPLOYED == 365243, NA, DAYS_EMPLOYED),
         DAYS_EMPLOYED_PERC = sqrt(DAYS_EMPLOYED / DAYS_BIRTH),
         INCOME_CREDIT_PERC = AMT_INCOME_TOTAL / AMT_CREDIT,
         INCOME_PER_PERSON = log1p(AMT_INCOME_TOTAL / CNT_FAM_MEMBERS),
         ANNUITY_INCOME_PERC = sqrt(AMT_ANNUITY / (1 + AMT_INCOME_TOTAL)),
         LOAN_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL,
         ANNUITY_LENGTH = AMT_CREDIT / AMT_ANNUITY,
         CHILDREN_RATIO = CNT_CHILDREN / CNT_FAM_MEMBERS, 
         CREDIT_TO_GOODS_RATIO = AMT_CREDIT / AMT_GOODS_PRICE,
         INC_PER_CHLD = AMT_INCOME_TOTAL / (1 + CNT_CHILDREN),
         SOURCES_PROD = EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3,
         CAR_TO_BIRTH_RATIO = OWN_CAR_AGE / DAYS_BIRTH,
         CAR_TO_EMPLOY_RATIO = OWN_CAR_AGE / DAYS_EMPLOYED,
         PHONE_TO_BIRTH_RATIO = DAYS_LAST_PHONE_CHANGE / DAYS_BIRTH,
         PHONE_TO_EMPLOY_RATIO = DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED) 

docs <- str_subset(names(tr), "FLAG_DOC")
live <- str_subset(names(tr), "(?!NFLAG_)(?!FLAG_DOC)(?!_FLAG_)FLAG_")
inc_by_org <- tr_te %>% 
  group_by(ORGANIZATION_TYPE) %>% 
  summarise(m = median(AMT_INCOME_TOTAL)) %$% 
  setNames(as.list(m), ORGANIZATION_TYPE)

rm(tr, te, fn, sum_bureau, sum_cc_balance, 
   sum_payments, sum_pc_balance, sum_prev); gc()

tr_te %<>% 
  mutate(DOC_IND_KURT = apply(tr_te[, docs], 1, moments::kurtosis),
         LIVE_IND_SUM = apply(tr_te[, live], 1, sum),
         NEW_INC_BY_ORG = recode(tr_te$ORGANIZATION_TYPE, !!!inc_by_org),
         NEW_EXT_SOURCES_MEAN = apply(tr_te[, c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")], 1, mean),
         NEW_SCORES_STD = apply(tr_te[, c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")], 1, sd))%>%
  mutate_all(funs(ifelse(is.nan(.), NA, .))) %>% 
  mutate_all(funs(ifelse(is.infinite(.), NA, .))) %>% 
  data.matrix()

cat("Preparing data...\n")
data <- tr_te[tri, ]
nrow(data)
nrow(Target)
data<-cbind(data,Target)
#data is the "Original Data" file we refer to in the paper with 743 features

#####DATA SAMPLING PROCEDURE FROM ORIGINAL DATA####

#import csvs of random rowIDs generated in Wolfram Mathematic for undersampling
femalerowID <- read.csv(file= "FemaleRowIDs.csv", header=FALSE, sep=",",na.strings=c("","NA"))
malerowID <- read.csv(file= "MaleRowIDs.csv", header=FALSE, sep=",",na.strings=c("","NA"))
#remove 4 XNA observations in CODE_GENDER
datasex<- subset(data, CODE_GENDER!= 3)
rm(data)

####TRAINING/TESTING SPLIT####
#split 80/20 train test
inTrain <- createDataPartition(y = datasex$Target, p = 0.8, list = FALSE) 
train <- datasex[ inTrain,]
test <- datasex[ -inTrain,]
rm(datasex)

####MINORITY DATA GENERATION####
#males==2, females==1
trainmale <- train[train$CODE_GENDER == 2, ]
trainfemale <- train[train$CODE_GENDER == 1, ]
#add count from 1 to 161,732 to trainfemale
trainfemale$RowNumber <- seq.int(nrow(trainfemale))

#sample 21,068 females from trainfemale (161,732) using femalerowID V1
femaleIDMIN <- data.frame(femalerowID$V1);names(femaleIDMIN)[1] <- "RowNumber"
trainMINfemale<- merge(femaleIDMIN,trainfemale, by="RowNumber")
trainMINfemale<- trainMINfemale[-c(1)]
trainMIN <- rbind.data.frame(trainmale,trainMINfemale)

#MODEL 1 MINORITY runs on:
#trainMIN

#MODEL 2 MINORITY runs on
trainnosexMIN <- trainMIN[-c(4)]

####REBALANCED DATA GENERATION####
#rebalancing procedure
#add count from 1 to 84,274 to trainmale
trainmale$RowNumber <- seq.int(nrow(trainmale))
maleIDMIN <- data.frame(malerowID$V1);names(maleIDMIN)[1] <- "RowNumber"
train50male<- merge(maleIDMIN,trainmale, by="RowNumber")
train50male<- train50male[-c(1)]
train50 <- rbind.data.frame(train50male,trainMINfemale)

#MODEL 1 REBALANCED runs on:
#train50

#MODEL 2 REBALANCED runs on
trainnosex50 <- train50[-c(4)]


####PREDICTIONS USING GRADIENT BOOSTING####

####MINORITY MODEL 1 (WITH GENDER)####
train<-trainMIN
Testing_Predictions<-test$Target
test$Target<-NULL
#Removing the gender from the test data
test$CODE_GENDER<-NULL

y=train$Target

train_1<-train%>%
  select(-Target)
cat("Preparing data...\n")

train_1<-as.matrix(train_1)
test1<-as.matrix(test)
dtest <- xgb.DMatrix(test1)
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = train_1[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = train_1[-tri, ], label = y[-tri])

cat("Training model...\n")
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 4,
          eta = 0.05,
          max_depth = 6,
          min_child_weight = 30,
          gamma = 0,
          subsample = 0.85,
          colsample_bytree = 0.7,
          colsample_bylevel = 0.632,
          alpha = 0,
          lambda = 0,
          nrounds = 2000)

set.seed(0)
m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 300)

#predictions
Target2<-predict(m_xgb,dtest)

Final_df<-cbind(test$SK_ID_CURR,Target2)
write.csv(Final_df,"Predictions_MinorityModel1.csv")

#AUC Calculation
model1MIN_ROC_prediction <- prediction(Final_df$Target2, test$Target)
auc.tmp <- performance(model1MIN_ROC_prediction,"auc") #Create AUC data
model1MIN_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
model1MIN_AUC

####MINORITY MODEL 2 (WITHOUT GENDER)####
train<-trainnosexMIN

y=train$Target

train_1<-train%>%
  select(-Target)
cat("Preparing data...\n")

train_1<-as.matrix(train_1)
test1<-as.matrix(test)
dtest <- xgb.DMatrix(test1)
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = train_1[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = train_1[-tri, ], label = y[-tri])

cat("Training model...\n")
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 4,
          eta = 0.05,
          max_depth = 6,
          min_child_weight = 30,
          gamma = 0,
          subsample = 0.85,
          colsample_bytree = 0.7,
          colsample_bylevel = 0.632,
          alpha = 0,
          lambda = 0,
          nrounds = 2000)

set.seed(0)
m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 300)

#predictions
Target2<-predict(m_xgb,dtest)
Final_df<-cbind(test$SK_ID_CURR,Target2)
write.csv(Final_df,"Predictions_MinorityModel2.csv")

#AUC Calculation
model2MIN_ROC_prediction <- prediction(Final_df$Target2, test$Target)
auc.tmp <- performance(model2MIN_ROC_prediction,"auc") #Create AUC data
model2MIN_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
model2MIN_AUC


####REBALANCED MODEL 1 (WITH GENDER)####
train<-train50

y=train$Target

train_1<-train%>%
  select(-Target)
cat("Preparing data...\n")

train_1<-as.matrix(train_1)
test1<-as.matrix(test)
dtest <- xgb.DMatrix(test1)
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = train_1[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = train_1[-tri, ], label = y[-tri])

cat("Training model...\n")
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 4,
          eta = 0.05,
          max_depth = 6,
          min_child_weight = 30,
          gamma = 0,
          subsample = 0.85,
          colsample_bytree = 0.7,
          colsample_bylevel = 0.632,
          alpha = 0,
          lambda = 0,
          nrounds = 2000)

set.seed(0)
m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 300)

#Predictions
Target2<-predict(m_xgb,dtest)
Final_df<-cbind(test$SK_ID_CURR,Target2)
write.csv(Final_df,"Predictions_RebalancedModel1.csv")

#AUC Calculation
model1REB_ROC_prediction <- prediction(Final_df$Target2, test$Target)
auc.tmp <- performance(model1REB_ROC_prediction,"auc") #Create AUC data
model1REB_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
model1REB_AUC


####REBALANCED MODEL 2 (WITHOUT GENDER)####
train<-trainnosex50

y=train$Target

train_1<-train%>%
  select(-Target)
cat("Preparing data...\n")

train_1<-as.matrix(train_1)
test1<-as.matrix(test)
dtest <- xgb.DMatrix(test1)
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = train_1[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = train_1[-tri, ], label = y[-tri])

cat("Training model...\n")
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 4,
          eta = 0.05,
          max_depth = 6,
          min_child_weight = 30,
          gamma = 0,
          subsample = 0.85,
          colsample_bytree = 0.7,
          colsample_bylevel = 0.632,
          alpha = 0,
          lambda = 0,
          nrounds = 2000)

set.seed(0)
m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 300)

#predictions
Target2<-predict(m_xgb,dtest)
Final_df<-cbind(test$SK_ID_CURR,Target2)
write.csv(Final_df,"Predictions_RebalancedModel2.csv")

#AUC Calculation
model2REB_ROC_prediction <- prediction(Final_df$Target2, test$Target)
auc.tmp <- performance(model2REB_ROC_prediction,"auc") #Create AUC data
model2REB_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
model2REB_AUC

