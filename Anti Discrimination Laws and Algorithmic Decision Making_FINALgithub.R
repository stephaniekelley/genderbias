####ANTI-DISCRIMINATION LAWS, AI, AND GENDER BIAS: A CASE STUDY IN NON-MORTGAGE FINTECH LENDING####
#Authors: Stephanie Kelley and Anton Ovchinnikov
#Special thanks to our Research Assistant Harshdeep Singh 
#last updated November 2021

#ensure necessary packages are installed
if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not
pacman::p_load("tidyverse", "ROCR", "caret", "xgboost", "magrittr", "pROC","ggplot2",
               "SHAPforxgboost","fastDummies", "rockchalk", "dlookr", "pscl", "compareDF", "forcats", "data.table") #Check, and if needed install the necessary packages

library(tidyverse)
library(ROCR)
library(caret)
library(xgboost)
library(magrittr)
library(pROC)
library(ggplot2)
library(SHAPforxgboost)
library(fastDummies)
library(rockchalk)
library(dlookr)
library(pscl)
library(compareDF)
library(forcats)
library(data.table)


#set a random number seed for use throughout analysis
set.seed(0)

options(max.print = 1000)

####4: Data, sampling, and analytical approach####
#datasets can be found at the Home Credit Default Risk competition on Kaggle
#website: https://www.kaggle.com/c/home-credit-default-risk as of November 2021
#import Kaggle data from the website and save files in  your workspace (files are too large to save on github)
#Disclaimer: the competition's rules prohibit using the data for research, however, Home Credit granted us explicit permission to use it for this study

####4.3.2: The machine learning process####
#Combine the datasets and perform feature engineering to create "Original Data"

#Load data from workspace
bbalance <- read_csv("bureau_balance.csv")
bureau  <- read_csv("bureau.csv")
cc_balance <- read_csv("credit_card_balance.csv")
payments <- read_csv("installments_payments.csv")
pc_balance <- read_csv("POS_CASH_balance.csv")
prev <- read_csv("previous_application.csv")
tr <- read_csv("application_train.csv")
te <- read_csv("application_test.csv")

#Preprocessing
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

#Preparing Kaggle Data for objective measurement in online competition
#Kaggle test data (not used in main analysis due to lack of target)
test <- tr_te[-tri, ]

#Kaggle train data - referred to as "Original Data" in our analyses
train <- tr_te[tri, ]
data<-cbind(train,Target)

#write data file to csv for records
write.csv(data, "originaldata.csv")

#re-import file, saved as csv in workspace
data <- read.csv(file= "originaldata.csv", header=TRUE, sep=",",na.strings=c("","NA"))

#random rowIDs generated in Mathematic for sampling procedures
#we have opted to use random numbers generated in Mathematica for replicability - these numbers are saved in csv files on github (https://github.com/stephaniekelley/genderbias)
testrowID <- read.csv(file= "testRowIDs.csv", header=TRUE, sep=",",na.strings=c("","NA"))
trainrowID <- read.csv(file= "trainRowIDs.csv", header=TRUE, sep=",",na.strings=c("","NA"))
femalerowID <- read.csv(file= "FemaleRowIDs.csv", header=FALSE, sep=",",na.strings=c("","NA"))
malerowID <- read.csv(file= "MaleRowIDs.csv", header=FALSE, sep=",",na.strings=c("","NA"))
trainREBfemalerowID <- read.csv(file= "trainREBcollfemaleID.csv", header=FALSE, sep=",",na.strings=c("","NA"))
trainREBmalerowID <- read.csv(file= "trainREBcollmaleID.csv", header=FALSE, sep=",",na.strings=c("","NA"))

#data cleaning
#remove 4 XNA observations in CODE_GENDER
data<- subset(data, CODE_GENDER!= 3)

#create "Training data" and "Testing data" using 80%/20% Training/Testing split
#split 80/20 train test
test <- merge(testrowID, data, by="SK_ID_CURR")
train<- merge(trainrowID, data, by="SK_ID_CURR")
rm(data)

#create "Minority data" using undersampling procedure
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

#remove SK_ID_CURR,X values for ML modeling
trainMIN<- trainMIN[-c(1:2)]
#MODEL 1 MINORITY runs on:
#trainMIN
#MODEL 2 MINORITY runs on
trainnosexMIN <- trainMIN[-c(2)]

####4.3.1: The traditional statistical modeling process####
#Combine training minority and testing datasets from main analysis to perform standard methodology for LR lending models, used by Andreeva and Matuszyk (2019)
trainID <- train[c(3)]
testID <- test[c(3)]
data <- bind_rows(train, test)
rm(train,test);gc()

#remove all engineered features from combined dataset
datalogfeatures<- data[c(3:123,743)]

#data to transform to factors
factorinfrequent <- function(x) {
  fct_infreq(as.factor(x))
} 

datafactors <- datalogfeatures %>% mutate_at(c(2:5,11:15,22:28,30:40,86:87,89:94,96:122), factorinfrequent)

rm(datalogfeatures);gc()

#bin numeric features - 10-20 bins to start with most frequent bucket as reference category 
#use binning from dlookr package https://www.rdocumentation.org/packages/dlookr/versions/0.3.13/topics/binning 
#create pretty binning function that sets factors base level to most frequent bucket
prettybin10<- function(x) {
  binning(x,10,ordered=TRUE, approxy.lab=FALSE)
} 

databinned <- datafactors %>% mutate_at(c(6:10,16:21,29,41:85,88,95), prettybin10)

summary(databinned$OBS_60_CNT_SOCIAL_CIRCLE)

#combine factors with empty levels, and similar default rates
#first determine with factors have empty levels
datacombined <- databinned %>% mutate(OBS_30_CNT_SOCIAL_CIRCLE=combineLevels(OBS_30_CNT_SOCIAL_CIRCLE, levs=c("10", "11", "12", "13", "14",
                                                                                                              "15","16","17","18","19","20",
                                                                                                              "21","22","23","24","25","26","27",
                                                                                                              "29","30","47","348"), newLabel=c("10+"))) %>%
  mutate(DEF_30_CNT_SOCIAL_CIRCLE=combineLevels(DEF_30_CNT_SOCIAL_CIRCLE, levs=c("4", "5", "6", "8", "34"), newLabel=c("4+"))) %>%
  mutate(OBS_60_CNT_SOCIAL_CIRCLE=combineLevels(OBS_60_CNT_SOCIAL_CIRCLE, levs=c("10", "11", "12", "13", "14",
                                                                                 "15","16","17","18","19","20",
                                                                                 "21","22","23","24","25","27",
                                                                                 "29","47","344"), newLabel=c("10+"))) %>%
  mutate(DEF_60_CNT_SOCIAL_CIRCLE=combineLevels(DEF_60_CNT_SOCIAL_CIRCLE, levs=c("4", "5", "6", "24"), newLabel=c("4+")))%>%
  mutate(AMT_REQ_CREDIT_BUREAU_DAY=combineLevels(AMT_REQ_CREDIT_BUREAU_DAY, levs=c("4", "5", "6", "8", "9"), newLabel=c("4+")))%>%
  mutate(AMT_REQ_CREDIT_BUREAU_WEEK=combineLevels(AMT_REQ_CREDIT_BUREAU_WEEK, levs=c("4", "5", "6", "7","8"), newLabel=c("4+")))%>%
  mutate(AMT_REQ_CREDIT_BUREAU_MON=combineLevels(AMT_REQ_CREDIT_BUREAU_MON, levs=c("10", "11", "12", "13","14","15","16","17","18","19",
                                                                                   "23","24","27"), newLabel=c("10+")))%>%
  mutate(AMT_REQ_CREDIT_BUREAU_QRT=combineLevels(AMT_REQ_CREDIT_BUREAU_QRT, levs=c("4", "5", "6", "7","8"), newLabel=c("4+")))%>%
  mutate(AMT_REQ_CREDIT_BUREAU_YEAR=combineLevels(AMT_REQ_CREDIT_BUREAU_YEAR, levs=c("10", "11", "12", "13","14","15","16","17","18","19",
                                                                                     "21","22", "23","25"), newLabel=c("10+")))%>%
  mutate(NAME_FAMILY_STATUS=combineLevels(NAME_FAMILY_STATUS, levs=c("5","6"), newLabel=c("5+")))

summary(datacombined$OBS_60_CNT_SOCIAL_CIRCLE)

summary(datacombined$EXT_SOURCE_1)


rm(databinned);gc()

#then coarse classes are transformed into binary dummy variables (Lin et al 2012),and original variables are removed
datadummy <- dummy_cols(datacombined, select_columns = c("CNT_CHILDREN","AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                                                         "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
                                                         "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2",
                                                         "EXT_SOURCE_3", "DAYS_LAST_PHONE_CHANGE","APARTMENTS_AVG","BASEMENTAREA_AVG",
                                                         "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG",
                                                         "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG",
                                                         "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",
                                                         "APARTMENTS_MODE","BASEMENTAREA_MODE",
                                                         "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE",
                                                         "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE",
                                                         "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE",
                                                         "APARTMENTS_MEDI","BASEMENTAREA_MEDI",
                                                         "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI",
                                                         "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",
                                                         "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",
                                                         "TOTALAREA_MODE","DAYS_LAST_PHONE_CHANGE"), ignore_na = FALSE, remove_most_frequent_dummy = TRUE, remove_selected_columns=TRUE)
datadummy <- datadummy %>% mutate_if(is.integer, as.factor)
datazeros <- datadummy %>% mutate_if(is.factor, fct_explicit_na, na_level="0")

rm(datadummy, databinned, datacombined); gc()

#resplit data into train MIN and test datasets with the same rows as the main analysis
trainbinned <- trainID %>% left_join(datazeros, by = "SK_ID_CURR")
testbinned <- testID %>% left_join(datazeros, by = "SK_ID_CURR")
rm(datazeros);gc()
summary(trainbinned$CODE_GENDER)



#create "Rebalanced Collected Data" using "extra" women observations discarded during creating of "Minority Data"
#sample an additional 31,603 from the original data (as if you gathered more data from customers or other sources)
names(trainREBfemalerowID)[1] <- "RowNumber"
trainREBcollfemale<- merge(trainREBfemalerowID,trainfemale, by="RowNumber")
trainREBcollfemale<- trainREBcollfemale[-c(1)]
#sample out 52,671 males from the minority dataset
trainmale$RowNumber <- seq.int(nrow(trainmale))
names(trainREBmalerowID)[1] <- "RowNumber"
trainREBcollmale <- merge(trainREBmalerowID,trainmale, by="RowNumber")
trainREBcollmale<- trainREBcollmale[-c(1)]
#combine 31,603 females to original 21,068 from minority sample, add in subset of males for a balanced 50/50 dataset
trainREBcoll <- rbind.data.frame(trainREBcollmale,trainMINfemale, trainREBcollfemale)
#remove SK_ID_CURR,X for modelling
trainREBcoll<- trainREBcoll[-c(1:2)]
#REBcoll MIN M1 runs on 
#trainREBcoll
#REB MIN M2 runs on
trainnosexREBcoll <- trainREBcoll[-c(2)]

#create "Rebalanced Data" using undersampling procedure
#add count from 1 to 84,274 to trainmale
trainmale$RowNumber <- seq.int(nrow(trainmale))
maleIDMIN <- data.frame(malerowID$V1);names(maleIDMIN)[1] <- "RowNumber"
train50male<- merge(maleIDMIN,trainmale, by="RowNumber")
train50male<- train50male[-c(1)]
train50 <- rbind.data.frame(train50male,trainMINfemale)
#remove SK_ID_CURR,X for modelling
train50<- train50[-c(1:2)]
#MODEL 1 REBALANCED runs on:
#train50
#MODEL 2 REBALANCED runs on
trainnosex50 <- train50[-c(2)]

#print to csv for records
write.csv(trainMIN, "trainMIN.csv")
write.csv(trainnosexMIN, "trainnosexMIN.csv")
write.csv(trainREBcoll, "trainREBcoll.csv")
write.csv(trainnosexREBcoll, "trainnosexREBcoll.csv")
write.csv(train50, "train50.csv")
write.csv(trainnosex50, "trainnosex50.csv")
write.csv(test, "test.csv")
rm(femaleIDMIN, femalerowID, maleIDMIN, malerowID, test, testrowID, train, train50, train50male, trainfemale, trainmale, trainMIN, trainnosexMIN,
   trainMINfemale, trainnosex50, trainnosexREBcoll, trainREBcoll, trainREBcollfemale, trainREBcollmale, trainREBfemalerowID,
   trainREBmalerowID, trainrowID);gc()

###4.2 Sampling: reject inference - augmentation####
#perform the augmentation sampling procedure to reduce selection bias in the testing dataset
#read in test dataset (if not already in workspace)
test <- read.csv("test.csv")

#create a joint distribution table of the top non-financial variables from the Home Credit testing data

#clean data as needed to create joint distribution table
#CODE_GENDER (Men = 2, women= 1)
#DAYS_BIRTH (clients age in days at the time of the application)

#change CODE_GENDER to factor for naming ease
table(test$CODE_GENDER)
test$CODE_GENDER <- as.factor(test$CODE_GENDER)
str(test$CODE_GENDER)
levels(test$CODE_GENDER) <- c("women", "men")

#change DAYS_BIRTH to age in years, binned into 10 groups
test$DAYS_BIRTH <- abs(test$DAYS_BIRTH)
test <- test %>% mutate(YEARS_BIRTH= DAYS_BIRTH*0.002738)
#bin numeric features - 10 bins to start with most frequent bucket as reference category 
#use binning from dlookr package https://www.rdocumentation.org/packages/dlookr/versions/0.3.13/topics/binning 
#create pretty binning function that sets factors base level to most frequent bucket
test$YEARS_BIRTH_BIN <- binning(test$YEARS_BIRTH, 10, type=c("pretty"), ordered=TRUE)
table(test$YEARS_BIRTH_BIN)

#Add NAME_FAMILY_STATUS (1= civil marriage, 2=married, 3=separated, 4=single, 5= unknown, 6=widow)
summary(test$NAME_FAMILY_STATUS)
test$NAME_FAMILY_STATUS <- as.factor(test$NAME_FAMILY_STATUS)
levels(test$NAME_FAMILY_STATUS) <- c("civilmarriage", "married", "separated","single", "unknown", "widow")
table(test$NAME_FAMILY_STATUS)

#AMT_INCOME_TOTAL ***Income of the client (WHAT CURRENCY? Checking with HC)
summary(test$AMT_INCOME_TOTAL)
test$AMT_INCOME_TOTAL <- binning(test$AMT_INCOME_TOTAL, nbins=4, type=c("quantile"), ordered=TRUE)
table(test$AMT_INCOME_TOTAL)

#OCCUPATION_TYPE
summary(test$OCCUPATION_TYPE)
test$OCCUPATION_TYPE <- as.factor(test$OCCUPATION_TYPE)
levels(test$OCCUPATION_TYPE) <- c("Accountants", "Cleaning staff", "Cooking staff","Core staff", "Drivers", "High skill tech staff",
                                  "HR staff", "IT staff", "Laborers","Low-skill Laborers", "Managers", "Medicine staff",
                                  "Private service staff", "Realty agents", "Sales staff","Secretaries", "Security staff",
                                  "Waiters/barmen staff","unknown")
test$OCCUPATION_TYPE[is.na(test$OCCUPATION_TYPE)]= "unknown"
test <- test %>% mutate(OCCUPATION_TYPE=fct_collapse(OCCUPATION_TYPE,
                                                     other= c("Private service staff", "Realty agents",
                                                              "Drivers", "IT staff", "Cooking staff", "Cleaning staff", 
                                                              "Low-skill Laborers", "Waiters/barmen staff")))
test<- test %>% mutate(OCCUPATION_TYPE = fct_relevel(OCCUPATION_TYPE, sort))
table(test$OCCUPATION_TYPE)

#TARGET
summary(test$Target)
table(test$Target)

#APPLICATION STATUS
#add application status to original data - all data is from approved borrowers, so status equals "approved"
test <- test %>% mutate(APPLICATION_STATUS= "approved")

#create joint distribution table
JD1 <- xtabs(~CODE_GENDER + YEARS_BIRTH_BIN + NAME_FAMILY_STATUS + AMT_INCOME_TOTAL + OCCUPATION_TYPE+ Target+ APPLICATION_STATUS, data=test)
JD1_DF <- as.data.frame(ftable(JD1))
prop.table(JD1)
JD1_DF_prop <- as.data.frame(prop.table(JD1))
write.csv(JD1_DF_prop, "Jointdistributiontable_test.csv")
rm(JD1,JD1_DF,JD1_DF_prop)

#read in the joint distribution table from the other bank (procedure discussed in detail in Section 4.2 in the paper)
UBP <- read.csv(file="Auto Loan Data_UBP_UNIV COLLAB.csv", header=TRUE, sep=",",na.strings=c("","NA"))

head(UBP)
#check missing
sapply(UBP, function(x) sum(is.na(x)))
head(JD1_DF)

#format "UBP" to match data from "test"

#CODE_GENDER
UBPnew <- UBP %>% rename(CODE_GENDER=Gender)
table(UBPnew$CODE_GENDER)
UBPnew$CODE_GENDER <- as.factor(UBPnew$CODE_GENDER)
str(UBPnew$CODE_GENDER)
levels(UBPnew$CODE_GENDER) <- c("women", "men", NA)
sum(is.na(UBPnew$CODE_GENDER))

#change DAYS_BIRTH to age in years, binned into 10 groups
head(UBPnew)
table(UBP$Age)
UBPnew <- UBPnew %>% rename(YEARS_BIRTH=Age)
summary(UBPnew$YEARS_BIRTH)
#clean data
UBPnew <- UBPnew %>% mutate(YEARS_BIRTH=ifelse(YEARS_BIRTH <0, NA, YEARS_BIRTH))%>%
  mutate(YEARS_BIRTH=ifelse(YEARS_BIRTH > 1900, 2021-YEARS_BIRTH, YEARS_BIRTH))%>%
  mutate(YEARS_BIRTH=ifelse(YEARS_BIRTH > 100, NA, YEARS_BIRTH))

#bin numeric features - 10 bins to start with most frequent bucket as reference category 
#use binning from dlookr package https://www.rdocumentation.org/packages/dlookr/versions/0.3.13/topics/binning 
#create pretty binning function that sets factors base level to most frequent bucket
table(UBPnew$YEARS_BIRTH)
UBPnew$YEARS_BIRTH_BIN <- binning(UBPnew$YEARS_BIRTH, 10, type=c("pretty"), ordered=TRUE)
table(UBPnew$YEARS_BIRTH_BIN)

#Add NAME_FAMILY_STATUS (1= civilmarriage, 2=married, 3=separated, 4=single, 5= unknown, 6=widow)
head(UBPnew)
table(UBP$Marital_Status)
#Annulled -"separated"
#Choose - "unknown"
#Divorced - "separated"
#Live-in - "civil marriage"
#Married- "married"
#N/A - "unknown"
#Separated -"separated"
#Single - "single"
#Sngl-Paren - "single)
#Unverified "unknown"
#Widow/er - "widow"
UBPnew <- UBPnew %>% rename(NAME_FAMILY_STATUS=Marital_Status)
UBPnew$NAME_FAMILY_STATUS <- as.factor(UBPnew$NAME_FAMILY_STATUS)
UBPnew <- UBPnew %>% mutate(NAME_FAMILY_STATUS=fct_collapse(NAME_FAMILY_STATUS,
                                                            civilmarriage= c("Live-in"),
                                                            married= c("Married"),
                                                            separated= c("Annulled","Divorced", "Separated"),
                                                            single=c("Single","Sngl-Paren"),
                                                            unknown=c("Choose","N/A","Unverified"),
                                                            widow=c("Widow/er")))
levels(UBPnew$NAME_FAMILY_STATUS) 
table(UBPnew$NAME_FAMILY_STATUS)


#AMT_INCOME_TOTAL 
head(UBPnew)
summary(UBPnew$Income)
UBPnew <- UBPnew %>% rename(AMT_INCOME_TOTAL=Income)
UBPnew$AMT_INCOME_TOTAL <- binning(UBPnew$AMT_INCOME_TOTAL, nbins=4, type=c("quantile"), ordered=TRUE)
table(test$AMT_INCOME_TOTAL)

#OCCUPATION_TYPE
UBPnew <- UBPnew %>% rename(OCCUPATION_TYPE=Occupation)
UBPnew <- UBPnew %>% mutate(OCCUPATION_TYPE=replace_na(OCCUPATION_TYPE, "unknown"))
UBPnew$OCCUPATION_TYPE <- as.factor(UBPnew$OCCUPATION_TYPE)
table(UBPnew$OCCUPATION_TYPE)
occupations <- as.data.frame(unique(UBPnew$OCCUPATION_TYPE))
write.csv(occupations, "occupations.csv")

#read in file of matched occupations (three research team members matched the occupation labels between the 2 banks to create a master list)
newoccupationnames <- read.csv(file= "occupations_matched.csv")
oldnames <- newoccupationnames$oldnames
newnames <- newoccupationnames$newnames
change <- UBPnew$OCCUPATION_TYPE
mappings <- setNames(oldnames, newnames)
args <- c(list(change),mappings)
str(args)
OCCUPATION_TYPE_NEW <- as.data.frame(do.call(fct_recode,args))
rm(oldnames, newnames, change, mappings, args)
UBPnew <- cbind(UBPnew, OCCUPATION_TYPE_NEW)
rm(OCCUPATION_TYPE_NEW,occupations, newoccupationnames)

UBPnew <- select(UBPnew,-OCCUPATION_TYPE)
UBPnew <- rename(UBPnew,"OCCUPATION_TYPE"=`do.call(fct_recode, args)`) 
levels(UBPnew$OCCUPATION_TYPE) 

UBPnew <- UBPnew %>% mutate(OCCUPATION_TYPE=fct_collapse(OCCUPATION_TYPE,
                                                         other= c("Private service staff", "Realty agents",
                                                                  "Drivers", "IT staff", "Cooking staff", "Cleaning staff", 
                                                                  "Low-skill Laborers", "Waiters/barmen staff")))
UBPnew <- UBPnew %>% mutate(OCCUPATION_TYPE = fct_relevel(OCCUPATION_TYPE, sort))
table(UBPnew$OCCUPATION_TYPE)


#TARGET
head(UBPnew)
UBPnew <- UBPnew %>% rename(Target=Loan_Default)
table(UBPnew$Target)

#APPLICATION STATUS
head(UBPnew)
UBPnew <- UBPnew %>% rename(APPLICATION_STATUS=Application_Status)
UBPnew$APPLICATION_STATUS <- as.factor(UBPnew$APPLICATION_STATUS)
levels(UBPnew$APPLICATION_STATUS)
UBPnew <- UBPnew %>% mutate(APPLICATION_STATUS=fct_collapse(APPLICATION_STATUS,
                                                            approved= c("APPROVED", "BOOKED"),
                                                            rejected= c("DECLINED"),
                                                            pending= c("IN-PROCESS","IN-PROCESS-RETURNED-TO-CPS-FROM-RRMD", "PENDING",
                                                                       "RETURNED-CPS","RETURNED-RRMD"),
                                                            cancelled = c("CANCELLED")))
levels(UBPnew$APPLICATION_STATUS) 
table(UBPnew$APPLICATION_STATUS)

#create dataframe with only approved and rejected applicants
UBP_A_R <- UBPnew %>% filter(str_detect(APPLICATION_STATUS,c("approved","rejected")))
UBP_A_R <- UBP_A_R%>% mutate(APPLICATION_STATUS=fct_collapse(APPLICATION_STATUS,
                                                             approved= c("approved","cancelled","pending"),
                                                             rejected= c("rejected")))
table(UBP_A_R$APPLICATION_STATUS)
JD2 <- xtabs(~CODE_GENDER + YEARS_BIRTH_BIN + NAME_FAMILY_STATUS + AMT_INCOME_TOTAL + OCCUPATION_TYPE+ Target+ APPLICATION_STATUS, data=UBP_A_R)
JD2_DF <- as.data.frame(ftable(JD2))
prop.table(JD2)
JD2_DF_prop <- as.data.frame(prop.table(JD2))
write.csv(JD2_DF_prop, "Jointdistributiontable_UBP_A_R.csv")

#create dataframe with only approved applicants
UBP_A <- UBP_A_R %>% filter(APPLICATION_STATUS == "approved")

JD3 <- xtabs(~CODE_GENDER + YEARS_BIRTH_BIN + NAME_FAMILY_STATUS + AMT_INCOME_TOTAL + OCCUPATION_TYPE+ Target+ APPLICATION_STATUS, data=UBP_A)
JD3_DF <- as.data.frame(ftable(JD3))
prop.table(JD3)
JD3_DF_prop <- as.data.frame(prop.table(JD3))
write.csv(JD3_DF_prop, "Jointdistributiontable_UBP_A.csv")

rm(JD2,JD2_DF,JD2_DF_prop, JD3, JD3_DF,JD3_DF_prop)

#compare approved/rejected dataframe to approved only dataframe
library(ggplot2)

#CODE_GENDER
ggplot(UBP_A_R, aes(x=Target, fill=CODE_GENDER))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted & Rejected")

ggplot(UBP_A, aes(x=Target, fill=CODE_GENDER))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted")

#remove NAs from both datasets for prop analysis
UBP_A_R_noNA <-UBP_A_R %>% drop_na(CODE_GENDER) 
UBP_A_noNA <-UBP_A %>% drop_na(CODE_GENDER)

ggplot(UBP_A_R_noNA, aes(x=Target, fill=CODE_GENDER))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted & Rejected")

ggplot(UBP_A_noNA, aes(x=Target, fill=CODE_GENDER))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted")

UBP_A_R_noNA%>% group_by(Target,CODE_GENDER)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))

UBP_A_noNA %>% group_by(Target,CODE_GENDER)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))

#test whether Gender proportions are different between Accepted & Reject dataset (UBP_A_R_noNA) and Accepted only (UBP_A_noNA)
#women who don't default
prop.test(x=c(21070,19352),n=c(nrow(UBP_A_R_noNA), nrow(UBP_A_noNA)))
#p-value = 0.2571 - not stat significantly different

#women who default
prop.test(x=c(577,572),n=c(nrow(UBP_A_R_noNA), nrow(UBP_A_noNA)))
#p-value = 0.0.2634 - not stat significantly different

#men who don't default
prop.test(x=c(31502,29137),n=c(nrow(UBP_A_R_noNA), nrow(UBP_A_noNA)))
#p-value = 0.7191 - not stat significantly different

#men who default
prop.test(x=c(3126,3092),n=c(nrow(UBP_A_R_noNA), nrow(UBP_A_noNA)))

#CODE_GENDER proportions are not different between accepts/reject data and accept only data

#NAME_FAMILY_STATUS#
ggplot(UBP_A_R, aes(x=Target, fill=NAME_FAMILY_STATUS))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted & Rejected")

ggplot(UBP_A, aes(x=Target, fill=NAME_FAMILY_STATUS))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted")

UBP_A_R%>% group_by(Target,NAME_FAMILY_STATUS)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))

UBP_A %>% group_by(Target,NAME_FAMILY_STATUS)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))

#test whether Family Status Proportions are different between A_R and A
#separated who don't default
prop.test(x=c(682,681),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.1877 - not stat significantly different

#unknown who don't default
prop.test(x=c(6360,5507),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 5.835e-05 - stat sign***

#look at only NAME_FAMILY_STATUS
#NAME_FAMILY_STATUS

UBP_A_R%>% group_by(NAME_FAMILY_STATUS)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))

UBP_A %>% group_by(NAME_FAMILY_STATUS)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))

#separated
prop.test(x=c(747,746),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.1662- not stat significantly different

#unknown
prop.test(x=c(8901,8030),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = p-value = 0.04099- stat sign***
#prop 1  0.1518268   prop 2 0.1474693
0.1474693/0.1518268
#0.9712995 - decrease in unknown in accepted

#civil marriage
prop.test(x=c(626,559),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.5154- not stat significantly different

#married
prop.test(x=c(33108,30834),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.6087- not stat significantly different

#single
prop.test(x=c(14126,13220),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.4766- not stat significantly different

#widow
prop.test(x=c(1118,1063),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.5959- not stat significantly different

#NAME_FAMILY_STATUS proportions are not different between accepts/reject data and accept only data

#OCCUPATION
ggplot(UBP_A_R, aes(x=Target, fill=OCCUPATION_TYPE))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted & Rejected")


ggplot(UBP_A, aes(x=Target, fill=OCCUPATION_TYPE))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted")

UBP_A_R%>% group_by(OCCUPATION_TYPE)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))

UBP_A %>% group_by(OCCUPATION_TYPE)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))

#Core staff
prop.test(x=c(31135,29452),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.0009797 stat sign diff***

#Managers
prop.test(x=c(5109,4526),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = = 0.01584 stat sign diff***

#YEARS_BIRTH_BIN
ggplot(UBP_A_R, aes(x=Target, fill=YEARS_BIRTH_BIN))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted & Rejected")

ggplot(UBP_A, aes(x=Target, fill=YEARS_BIRTH_BIN))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted")

UBP_A_R%>% group_by(YEARS_BIRTH_BIN)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))

UBP_A %>% group_by(YEARS_BIRTH_BIN)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))

#[0,10]
prop.test(x=c(33,32),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.9605- not stat significantly different

#(10,20]
prop.test(x=c(147,145),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.6483- not stat significantly different

#(20,30]
prop.test(x=c(1166,1145),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.183- not stat significantly different

#(30,40]
prop.test(x=c(11039,10250),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.9868- not stat significantly different

#(40,50]
prop.test(x=c(16796,15372),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.1202- not stat significantly different

#(50,60]
prop.test(x=c(13906,12830),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.5372- not stat significantly different

#(60,70]
prop.test(x=c(7238,6634),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.4094- not stat significantly different

#(70,80]
prop.test(x=c(1537,1424),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.9598- not stat significantly different

#(80,90]
prop.test(x=c(188,182),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.7287- not stat significantly different

#(90,100]
prop.test(x=c(16,16),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = 0.9744- not stat significantly different

#check age again on non-binned variable
#use two sample Kolmogorov-Smirnov Tests
ks.test(UBP_A$YEARS_BIRTH, UBP_A_R$YEARS_BIRTH)
#p-value = 0.8885- not stat significantly different

#AMT_INCOME_TOTAL

ggplot(UBP_A_R, aes(x=Target, fill=AMT_INCOME_TOTAL))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted & Rejected")

ggplot(UBP_A, aes(x=Target, fill=AMT_INCOME_TOTAL))+
  geom_bar(position="fill")+
  labs(y="Proportion", title= "UBP Accepted")

UBP_A_R%>% group_by(AMT_INCOME_TOTAL)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))


UBP_A %>% group_by(AMT_INCOME_TOTAL)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))


#(1,3.6e+04]
prop.test(x=c(13542, 11242),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = < 2.2e-16- stat sign different ****
#  prop 1  0.2309897  prop 2  0.2064571
0.2064571/0.2309897

#(3.6e+04,6e+04]
prop.test(x=c(15233, 14770),n=c(nrow(UBP_A_R), nrow(UBP_A)))
#p-value = < 2.2e-16- stat sign different ****
#  prop 1  0.2309897  prop 2  0.2064571
0.2064571/0.2309897

Prop_A_R <- UBP_A_R %>% group_by(AMT_INCOME_TOTAL, OCCUPATION_TYPE, .drop=FALSE)%>%
  summarise(n=n()) %>%
  mutate(prop = n/nrow(UBP_A_R))%>%
  mutate(status = "AR")

Prop_A <- UBP_A %>% group_by(AMT_INCOME_TOTAL,OCCUPATION_TYPE, .drop=FALSE)%>%
  summarise(n=n()) %>%
  mutate(prop = n/nrow(UBP_A))%>%
  mutate(status = "A")


#create vectors of proportions
prop_A_R <- UBP_A_R %>% group_by(AMT_INCOME_TOTAL, OCCUPATION_TYPE, .drop=FALSE)%>%
  summarise(n=n()) %>%
  mutate(prop = n/nrow(UBP_A_R))%>%
  mutate(status = "AR")%>%
  pull(prop)

prop_A <- UBP_A %>% group_by(AMT_INCOME_TOTAL,OCCUPATION_TYPE, .drop=FALSE)%>%
  summarise(n=n()) %>%
  mutate(prop = n/nrow(UBP_A))%>%
  mutate(status = "A")%>%
  pull(prop)


#create proportion ratio vector
propratio <- prop_A_R/prop_A


#find category with highest increase rate
max(propratio)
#1.578965
#find details for category with highest increase rate
Prop_A[which(propratio == max(propratio)),]
#(1.2e+05,1.1e+18] - last income group, and "other"

#remove 1 row from test group with Income = NA
test <- test %>% drop_na(AMT_INCOME_TOTAL)

#find size of that group in test ((2.02e+05,1.8e+07], Other) - last income bucket

testfreq[44,3]

#highest growth rate bucket is 2164

2164/61500
#and it represents 0.03518699 of the original test dataset
2164/0.03518699
2164/(0.03518699*1.578965)

#upweight size of that bucket
1.578965*2164
# = 3416.88
#downsampling ratio
2164/3416.88
#=0.6333263

testfreq <- test %>% group_by(AMT_INCOME_TOTAL,OCCUPATION_TYPE)%>%
  summarise(n=n()) %>%
  pull(n)
sum(testfreq)

upweights <- testfreq * propratio

downsamplefactor <- testfreq[44]/upweights[44]

downweights <- round(upweights * downsamplefactor)

sum(downweights)

rm(UBP_A_R, UBP_A, Prop_A_R, Prop_A, UBP, UBPnew)


#select downsampling weights
weights <- downweights

categories  <- test %>% group_by(AMT_INCOME_TOTAL,OCCUPATION_TYPE)%>%
  summarise(n=n()) %>%
  select(-n)

testA <- test
testA[,"frq"]<- 0

#fill in target frq for each factor combination of Income & Occupation
for (i in 1:length(testfreq)){                                                                       
  Income <- as.vector(categories[[i,1]]) 
  Occupation <- as.vector(categories[[i,2]])
  weight <- weights[i]
  for(r in 1:nrow(testA)){
    if(as.vector(testA$AMT_INCOME_TOTAL[r]) == Income && as.vector(testA$OCCUPATION_TYPE[r]) == Occupation){
      testA[r,"frq"] <- weight
    } else {
      testA[r,"frq"] <- testA$frq[r]
    }
  }
}

rm(test,testfreq, propratio, weights, categories, Income, Occupation, r, i, weight)

#check frequencies were assigned correctly
testA %>% group_by(AMT_INCOME_TOTAL, OCCUPATION_TYPE)%>%
  summarise(frq=min(frq))

#split the data to prepare for stratified sampling
s <- split(testA, list(testA$AMT_INCOME_TOTAL, testA$OCCUPATION_TYPE))
#resample
newtest <- lapply(s, function(x) sample_n(x, size=unique(x$frq), replace=FALSE))%>%
  do.call(what = rbind)

#check new sample
newtest %>% group_by(AMT_INCOME_TOTAL, OCCUPATION_TYPE)%>%
  summarise(n=n()) %>%
  mutate(prop = n/sum(n))                                                  

#remove frq column from new test
newtest<- newtest %>% select(-frq)
ARtest <- newtest

#check unique IDs after sampling procedure is complete
length(unique(ARtest$SK_ID_CURR))


#save new ARtest SK_ID_CURR list as new dataset
write.csv(as.data.frame(ARtest$SK_ID_CURR), "ARtest.csv")

rm(test, Incomes, s, testA)




####6.2.2. Machine Learning Discrimination: Gender-blind Feature Selection####
#train XGBoost tree ensemble on Minority data with gender (Model 1) to understand feature importance
train <- read.csv("trainMIN.csv")
test <- read.csv("test.csv")
Testing_Predictions<-test$Target
test$Target<-NULL

y=train$Target

train_1<-train%>%
  select(-Target)
cat("Preparing data...\n")

train_1<-as.matrix(train_1)
test1<-as.matrix(test)
dtest <- xgb.DMatrix(test1)

dtrain <- xgb.DMatrix(data = train_1, label = y)

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

m_xgbM1 <- xgb.train(p, dtrain, p$nrounds, print_every_n = 50)

#calculate SHAP values for explainability
shap_valuesM1 <- shap.values(xgb_model= m_xgbM1, X_train = train_1)
shap_valuesdfM1 <- as.data.frame(shap_valuesM1$mean_shap_score)
write.csv(shap_valuesdfM1, "shapvaluesM1.csv")

shap_longM1 <- shap.prep(xgb_model= m_xgbM1, X_train = train_1)
shap.plot.summary(shap_longM1)

#create  subset of dataset as required for explorations, given computation requirements of SHAP interaction values (Lundberg et al. 2019, https://arxiv.org/abs/1905.04610)
#dataset include gender-reliant features, gender-redundant features, and top 5 important gender-neutral features
condensed <- trainMIN %>% select(CODE_GENDER, EXT_SOURCE_2, EXT_SOURCE_3, ANNUITY_LENGTH, EXT_SOURCE_1, 
                                          CREDIT_TO_GOODS_RATIO, AMT_ANNUITY, NAME_EDUCATION_TYPE, DAYS_BIRTH,
                                          DAYS_LAST_DUE_1ST_VERSION_max, NAME_FAMILY_STATUS, CAR_TO_EMPLOY_RATIO,
                                          AMT_GOODS_PRICE, DAYS_ID_PUBLISH, ORGANIZATION_TYPE, 
                                          NAME_CASH_LOAN_PURPOSE_max, CNT_CREDIT_PROLONG_n_distinct, CNT_CREDIT_PROLONG_sum,
                                          AMT_INST_MIN_REGULARITY_min, RATE_INTEREST_PRIVILEGED_max, LIVE_REGION_NOT_WORK_REGION,
                                          NFLAG_LAST_APPL_IN_DAY_min, STATUS_min_n_distinct, 
                                          AMT_DRAWINGS_OTHER_CURRENT_sum, AMT_DRAWINGS_OTHER_CURRENT_max, CNT_DRAWINGS_CURRENT_min,
                                          CNT_DRAWINGS_OTHER_CURRENT_mean, CNT_DRAWINGS_POS_CURRENT_n_distinct, CREDIT_CURRENCY_sd,
                                          NAME_CONTRACT_STATUS_n_distinct.x, NAME_CONTRACT_TYPE_min, NFLAG_INSURED_ON_APPROVAL_min, 
                                          NFLAG_LAST_APPL_IN_DAY_sd, RATE_INTEREST_PRIVILEGED_min,
                                          Target) 

#train XGBoost tree ensemble on condensed dataset with gender (Model 1) to understand gender interactions

train <- condensed
test$Target<-NULL

y=train$Target

train_1<-train%>%
  select(-Target)
cat("Preparing data...\n")

train_1<-as.matrix(train_1)
test1<-as.matrix(test)
dtest <- xgb.DMatrix(test1)

dtrain <- xgb.DMatrix(data = train_1, label = y)

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

m_xgbtoyM1 <- xgb.train(p, dtrain, p$nrounds, print_every_n = 50)

#compute SHAP values and SHAP interaction values for condensed dataset
shap_valuesM1 <- shap.values(xgb_model= m_xgbtoyM1, X_train = train_1)
shap_valuesdfM1 <- as.data.frame(shap_valuesM1$mean_shap_score)
shap_scoredfM1 <- as.data.frame(shap_valuesM1$shap_score)
write.csv(shap_valuesdfM1, "shaptoymodelM1.csv")
write.csv(shap_scoredfM1, "shapscoretoyM1.csv")

#males =2, females=1
shap_scorewithgender <- cbind(shap_scoredfM1,trainMIN$CODE_GENDER);names(shap_scorewithgender)[35] <- "Gender"
shap_scorewithgender <- cbind(shap_scorewithgender, trainMIN$Target);names(shap_scorewithgender)[36]<- "Target"

shap_longM1 <- shap.prep(xgb_model= m_xgbtoyM1, X_train = train_1)
shap.plot.summary(shap_longM1)

shap_intM1 <- shap.prep.interaction(xgb_model= m_xgbtoyM1, X_train = train_1)
shap.plot.summary(shap_intM1)


#train XGBoost tree ensemble on Minority data  without gender (Model 2) to understand feature importance
train <- read.csv("trainnosexMIN.csv")
test <- read.csv("test.csv")
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

dtrain <- xgb.DMatrix(data = train_1, label = y)

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

m_xgbM2 <- xgb.train(p, dtrain, p$nrounds, print_every_n = 50)

#calculate SHAP values for explainability
shap_valuesM2 <- shap.values(xgb_model= m_xgbM2, X_train = train_1)
shap_valuesdfM2 <- as.data.frame(shap_valuesM1$mean_shap_score)
write.csv(shap_valuesdfM2, "shapvaluesM2.csv")

shap_longM2 <- shap.prep(xgb_model= m_xgbM2, X_train = train_1)
shap.plot.summary(shap_longM2)


#train XGBoost tree ensemble on condensed dataset without gender (Model 2) to understand gender interactions
condensednosex <- condensed %>% select(-CODE_GENDER)
train <- condensednosex
test$Target<-NULL

y=train$Target

train_1<-train%>%
  select(-Target)
cat("Preparing data...\n")

train_1<-as.matrix(train_1)
test1<-as.matrix(test)
dtest <- xgb.DMatrix(test1)

dtrain <- xgb.DMatrix(data = train_1, label = y)


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

m_xgbtoyM2 <- xgb.train(p, dtrain, p$nrounds, print_every_n = 50)

#get the feature names
names <- dimnames(data.matrix(train_1))[[2]]
#compute feature importance matrix
importance_matrix <- xgb.importance(names, model= m_xgbtoyM2)
#graph
xgb.plot.importance(importance_matrix[1:9,])

xgb.plot.tree(model= m_xgbtoyM2, trees=0, show_node_id = TRUE)

shap_valuesM2 <- shap.values(xgb_model= m_xgbtoyM2, X_train = train_1)
shap_valuesdfM2 <- as.data.frame(shap_valuesM2$mean_shap_score)
shap_scoredfM2 <- as.data.frame(shap_valuesM2$shap_score)
write.csv(shap_valuesdfM2, "shaptoymodelM2.csv")
write.csv(shap_scoredfM2, "shapscoretoyM2.csv")

shap_longM2 <- shap.prep(xgb_model= m_xgbtoyM2, X_train = train_1)
shap.plot.summary(shap_longM2)

shap_intM2 <- shap.prep.interaction(xgb_model= m_xgbtoyM2, X_train = train_1)
shap.plot.summary(shap_intM2)



####7. Possible Approaches to reduce discrimination####
#gender-aware hyperparameter tuning: creating a single models that has hyperparameters turning using gender of other applicants from Level 1 or 2 country
#model is then trained on rebalanced data without individual applicant gender (Model 2)
train <- read.csv("trainnosex50.csv")
test<-read.csv("test.csv")
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

dtrain <- xgb.DMatrix(data = train_1, label = y)

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

m_xgb <- xgb.train(p, dtrain, p$nrounds, print_ev01ery_n = 50)

#predictions
Target2<-predict(m_xgb,dtest)
Final_df<-cbind(test$SK_ID_CURR,Target2)
write.csv(Final_df,"Predictions_GenderawareHyperparameter.csv")



####Online Supplement 2: Impact of gender-exclusion on traditional statistical models####
#here we run the logistic regression model; the machine learning models are run in DataRobot

#logistic regression with gender (Model 1)
#per Andreeva and Matuszyk (2019) methodology variable must be statistically significant at 0.05 AND it much show "high" predictive power (AUC)
#per Andreeva and Matuszyk (2019) methodology "the variable that show significant statistical effect are consistent with general literature on credit scoring"
#run model on full training dataset (with 456 variables --> 122 features)
M1 <- glm(Target ~ . - SK_ID_CURR, data= trainbinned, family = binomial(link="logit"))
options(max.print=10000)
summary(M1) 

#testing individual feature AUCs to determine if they should be included in prediction
#generate smaller training subsets for coarse classified variables that have been converted to binary dummies
cntchildren<- train %>% select(starts_with('CNT_CHILDREN'), Target)
rm(fondkapremont)

M0<- glm(Target ~ . ,data= cntchildren, family = binomial(link="logit"))
summary(M0) 
prediction <- predict(M0,test, type= "response")
AUC_M0 <- auc(test$Target, prediction)
CI_M0 <- ci.auc(AUC_M0) #Confidence Interval for AUC - 2000 bootstraps for 95% CI
print(AUC_M0)
print(CI_M0)

#run model with features that have AUC CIs that cross 50% -per Andreeva and Matuszyk (2019) methodology
train1.1<- trainbinned %>% select(SK_ID_CURR, CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY,NAME_INCOME_TYPE, NAME_EDUCATION_TYPE,
                                  NAME_FAMILY_STATUS,NAME_HOUSING_TYPE,FLAG_WORK_PHONE, FLAG_PHONE, OCCUPATION_TYPE, REGION_RATING_CLIENT,
                                  REGION_RATING_CLIENT_W_CITY, WEEKDAY_APPR_PROCESS_START, HOUR_APPR_PROCESS_START,ORGANIZATION_TYPE,
                                  FONDKAPREMONT_MODE, WALLSMATERIAL_MODE,
                                  DEF_30_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE,FLAG_DOCUMENT_3, FLAG_DOCUMENT_5, FLAG_DOCUMENT_6,
                                  FLAG_DOCUMENT_8, FLAG_DOCUMENT_13, FLAG_DOCUMENT_15, FLAG_DOCUMENT_16, FLAG_DOCUMENT_18,
                                  AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR,
                                  starts_with('AMT_CREDIT_'),starts_with('AMT_ANNUITY_'),starts_with('AMT_GOODS_PRICE_'),
                                  starts_with('DAYS_BIRTH_'), starts_with('DAYS_EMPLOYED_'),starts_with('DAYS_REGISTRATION_'), 
                                  starts_with('DAYS_ID_PUBLISH_'),starts_with('OWN_CAR_AGE_'),starts_with('EXT_SOURCE_1_'),
                                  starts_with('EXT_SOURCE_2_'), starts_with('EXT_SOURCE_3_'), starts_with('DAYS_LAST_PHONE_CHANGE_'), 
                                  starts_with('APARTMENTS_AVG_'),starts_with('FLOORSMIN_AVG'), starts_with('LIVINGAREA_AVG_'),
                                  starts_with('BASEMENTAREA_MODE_'),
                                  starts_with('FLOORSMAX_MODE'),starts_with('LIVINGAREA_MODE'),starts_with('NONLIVINGAREA_MODE_'),
                                  starts_with('FLOORSMIN_MEDI_'),starts_with('LIVINGAREA_MEDI_'),starts_with('NONLIVINGAREA_MEDI_'),
                                  starts_with('TOTALAREA_MODE_'),Target)

#run final model with only significant features
M1.1 <- glm(Target ~. -SK_ID_CURR, data= train1.1, family = binomial(link="logit"))
summary(M1.1) 

#Pseudo-R squared statistics (McFadden - higher is better- between 0.2 to 0.4 very good)
pR2(M1.1) #0.1256

#review predictions
predictionMINM1 <- predict(M1.1,testbinned, type= "response")
LRAUC_MINM1 <- auc(testbinned$Target, predictionMINM1)
LRCI_MINM1 <- ci.auc(LRAUC_MINM1) #Confidence Interval for AUC - 2000 bootstraps for 95% CI
print(LRAUC_MINM1)
print(LRCI_MINM1)
write.csv(predictionMINM1, "Predictions_LRMINM1.csv")
head(testbinned)

#generate binary prediction at mean(predictionMINM1) cutoff
binarypredictionMINM1 <- ifelse(predictionMINM1 > mean(predictionMINM1),1,0)
print(confusionMatrix(as.factor(binarypredictionMINM1),as.factor(testbinned$Target), positive= "1", dnn=c("Predicted","Actual")))

#Final Model Fit
#plot residuals
plot(M1.1)

#check multicollinearity using the vif()
#function from the car package (anything greater than 7 is very high and likely an issue)
library(car)
vif(M1.1)
1/vif(M1.1)

#logistic regression without gender (Model 2), using same features as Model 1
M2 <- glm(Target ~. -SK_ID_CURR - CODE_GENDER, data= train1.1, family = binomial(link="logit"))
summary(M2) 

#Pseudo-R squared statistics (McFadden - higher is better- between 0.2 to 0.4 very good)
pR2(M2) #0.1141

#review predictions
predictionMINM2 <- predict(M2,testbinned, type= "response")
LRAUC_MINM2 <- auc(testbinned$Target, predictionMINM2)
LRCI_MINM2 <- ci.auc(LRAUC_MINM2) #Confidence Interval for AUC - 2000 bootstraps for 95% CI
print(LRAUC_MINM2)
print(LRCI_MINM2)

#generate binary prediction at mean(predictionMINM2) cutoff (different than M1)
binarypredictionMINM2 <- ifelse(predictionMINM2 > mean(predictionMINM2),1,0)
print(confusionMatrix(as.factor(binarypredictionMINM2),as.factor(testbinned$Target), positive= "1", dnn=c("Predicted","Actual")))

write.csv(predictionMINM2, "Predictions_LRMINM2.csv")

#Final Model Fit
#plot residuals
plot(M2)

#check multicollinearity using the vif()
#function from the car package (anything greater than 7 is very high and likely an issue)
library(car)
vif(M2)
1/vif(M2)

#compare models with and without features that have AUC CIs that cross 50%
anova(M1.1, M2, test= 'Chisq')
#M1 statistically significantly better

library(stargazer)
stargazer(M1.1, M2, type= "html", out="LogisticRegression_M1M2.doc", 
          intercept.bottom=F, intercept.top= T, digits=4)


####Online Supplement 3: A stylized example: generalizing the properties of gender-blind feature selection####
#data available at github: https://github.com/stephaniekelley/genderbias
style <- read_csv("AgedataV12.csv")
stylelog <- style %>% mutate(Target = as.factor(Target),
                             Gender = as.factor(Gender))
stylelog <- stylelog %>% mutate(Gender2 = ifelse(Gender==1,1,0))%>%
  mutate(Gender2 = as.factor(Gender2))

#1: Stylized LR (Model 1)
StyleMINM1 <- glm(Target~ Gender +  Income + Score +
                    Age + Gender*Age + Tenure, data= stylelog, family = binomial(link="logit"),na.action=na.exclude)
summary(StyleMINM1) 
coef(StyleMINM1)

#2: Stylized LR (Model 2)
StyleMINM2 <- glm(Target~  Income + Score+ 
                    Age  + Tenure , data= stylelog, family = binomial(link="logit"),na.action=na.exclude)
summary(StyleMINM2)
coef(StyleMINM2)

#assess the predicted probability of default of example woman and man generated by Model 1 and Model 2 - only difference is their gender

#Model 1 with gender 
#female, 100,000 income, EXT score of 320, Tenure of 20, Age 60 (XXX) #NO DEFAULT
fem60M1= data.frame(Income = 100000, Age = 60, Gender=1, Score = 360, Tenure = 20)
fem60M1 <- fem60M1 %>% mutate(Gender = as.factor(Gender))
predict(StyleMINM1, fem60M1, type="response")

#female, 100,000 income, EXT score of 320, Tenure of 20, Age 65 (0.1217718) #NO DEFAULT
fem65M1= data.frame(Income = 100000, Age = 65, Gender=1, Score = 360, Tenure = 20)
fem65M1 <- fem65M1 %>% mutate(Gender = as.factor(Gender))
predict(StyleMINM1, fem65M1, type="response")

#male, 100,000 income, EXT score of 320, Tenure of 20, Age 60 (XXX) #NO DEFAULT
mal60M1= data.frame(Income = 100000, Age = 60, Gender=2, Score = 360, Tenure = 20)
mal60M1 <- mal60M1 %>% mutate(Gender = as.factor(Gender))
predict(StyleMINM1, mal60M1, type="response")

#male, 100,000 income, EXT score of 320, Tenure of 20, Age 65 (0.1540111) #DEFAULTS
mal65M1= data.frame(Income = 100000, Age = 65, Gender=2, Score = 360, Tenure = 20)
mal65M1 <- mal65M1 %>% mutate(Gender = as.factor(Gender))
predict(StyleMINM1, mal65M1, type="response")


#Model 2 without gender
#female, 100,000 income, EXT score of 320, Tenure of 20, Age 60 (XXX) #STILL NO DEFAULT BUT RR UP 2X
fem60M2= data.frame(Income = 100000, Age = 60, Gender=1, Score = 360, Tenure = 20)
fem60M2 <- fem60M2 %>% mutate(Gender = as.factor(Gender))
predict(StyleMINM2, fem60M2, type="response")

#female, 100,000 income, EXT score of 320, Tenure of 20, Age 65 (0.142057) #NOW DEFAULTS (BAD)
fem65M2= data.frame(Income = 100000, Age = 65, Gender=1, Score = 360, Tenure = 20)
fem65M2 <- fem65M2 %>% mutate(Gender = as.factor(Gender))
predict(StyleMINM2, fem65M2, type="response")

#male, 100,000 income, EXT score of 320, Tenure of 20, Age 60 (XXX) #NO DEFAULT RR DECREASES
mal60M2= data.frame(Income = 100000, Age = 60, Gender=2, Score = 360, Tenure = 20)
mal60M2 <- mal60M2 %>% mutate(Gender = as.factor(Gender))
predict(StyleMINM2, mal60M2, type="response")

#male, 100,000 income, EXT score of 320, Tenure of 20, Age 65 (0.142057) #DEFAULTS BUT RR GOES DOWN
mal65M2= data.frame(Income = 100000, Age = 65, Gender=2, Score = 360, Tenure = 20)
mal65M2 <- mal65M2 %>% mutate(Gender = as.factor(Gender))
predict(StyleMINM2, mal65M2, type="response")


#test data with no missingness
#data available at github: https://github.com/stephaniekelley/genderbias
stylenomiss <- read_csv("AgedataV12nomiss.csv")
stylelognomiss <- stylenomiss %>% mutate(Target = as.factor(Target),
                                         Gender = as.factor(Gender))
stylelognomiss <- stylelognomiss %>% mutate(Gender2 = ifelse(Gender==1,1,0))%>%
  mutate(Gender2 = as.factor(Gender2))

#3: Stylized LR No Missingness (Model 1)
StyleMINM1nomiss <- glm(Target~ Gender +  Income + Score +
                          Age + Gender*Age + Tenure, data= stylelognomiss, family = binomial(link="logit"),na.action=na.exclude)
summary(StyleMINM1nomiss) 
coef(StyleMINM1nomiss)

#4: Stylized LR No Missingness (Model 2)
StyleMINM2nomiss <- glm(Target~  Income + Score+ 
                          Age  + Tenure , data= stylelognomiss, family = binomial(link="logit"),na.action=na.exclude)
summary(StyleMINM2nomiss)
coef(StyleMINM2nomiss)

library("stargazer")
stargazer(StyleMINM1, StyleMINM2,StyleMINM1nomiss, StyleMINM2nomiss, type= "html", out="style_binarylogistic_V12NoMiss_Aug18th.doc", 
          intercept.bottom=F, intercept.top= T, digits=5)


#5: Regular LR No Missing (replicate regular OVB)
StyleMINM1OVB <- glm(Target~ Gender +  Income + Score +
                       Age + Tenure, data= stylelognomiss, family = binomial(link="logit"),na.action=na.exclude)
summary(StyleMINM1OVB) 

#Tenure is removed because it is not significant
StyleMINM2OVB <- glm(Target~  Income + Score +
                       Age , data= stylelognomiss, family = binomial(link="logit"),na.action=na.exclude)
summary(StyleMINM2OVB)

