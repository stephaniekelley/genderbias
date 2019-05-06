if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} # Check if you have universal installer package, install if not
pacman::p_load("gbm", "ROCR", "caret", "onehot", "tidyverse", "naniar", "simputation", "glmnet", "modelr") #Check, and if needed install the necessary packages

library(gbm)
library(ROCR)
library(caret)
library(onehot)
library(tidyverse)
library(naniar)
library(simputation)
library(glmnet)
library(modelr)
set.seed(77850) #set a random number generation seed for us throughout analysis

#import data
#at the time of publishing this code, the data was publicly available available at
#https://www.kaggle.com/c/home-credit-default-risk/data, with the 
#dataset for the main analyses titled  “application_train.csv”. 

#import file, saved as csv in workspace
data <- read.csv(file= "application_train.csv", header=TRUE, sep=",",na.strings=c("","NA"))

####DATA PREPROCESSING####
#remove 4 XNA observations in CODE_GENDER
datasex<- subset(data, CODE_GENDER!= "XNA")
IDTARGET <- datasex[c(1:2)]
rm(data)

#categorical variable preprocessing

#code categorical variables as factors
datasex[,c(2,4:6,12:16,23:29,31:33,35:41,87,88,91,97:116)]<-lapply(datasex[,c(2,4:6,12:16,23:29,31:33,35:41,87,88,91,97:116)], as.factor)

#combine rare values of categorical variables for prediction stability
#Create a custom function to combine rare categories into "Other."
#+the name of the original variable (e.g., Other.State)
#This function has two arguments: the name of the dataframe and then
#count of observation in a category to define "rare"
combinerarecats<-function(data_frame,mincount){ 
  for (i in 1 : ncol(data_frame)){
    a<-data_frame[,i]
    replace <- names(which(table(a) < mincount))
    levels(a)[levels(a) %in% replace] <-paste("Rare",colnames(data_frame)[i],sep=".")
    data_frame[,i]<-a }
  return(data_frame) }

#combine categories with <100 values in data into "Other"
datasex<-combinerarecats(datasex,100) 

#onehot encoding of categorical variables
toencode <-datasex[,sapply(datasex,is.factor)]
toencode <- toencode[-c(1)]#remove target variable
encoder<- onehot(toencode, stringsAsFactors=TRUE, addNA=TRUE, max_levels = 20) #OCCUPATION_TYPE excluded for > max_levels
dataencoded<- as.data.frame(predict(encoder, toencode))
dataencoded[,c(1:198)]<-lapply(dataencoded[,c(1:198)], as.factor)
rm(encoder); gc()

#numerical variables

#median imputation 
toimpute <- datasex[,sapply(datasex,is.numeric)]; toimpute <- toimpute[-c(1)]
shadow <- as_shadow(toimpute)
dataimp <- impute_median_all(toimpute)
rm(toimpute)

#standardize numerical variables
stdnumerical <- as.data.frame(scale(dataimp, center=TRUE, scale=TRUE))
rm(dataimp)

#bind categorical variables with numerical variables
dataall <- cbind.data.frame(IDTARGET,dataencoded,stdnumerical, shadow)
dataglm <- cbind.data.frame(IDTARGET,toencode, stdnumerical)
rm(dataencoded, IDTARGET, shadow, stdnumerical,toencode); gc()

#remove variables without variation
datavar <- dataall[-c(5,8,9,12,15,23,29,35,42,49,50:52,55,58,61,64,67,86,90,94,102,105,108,111,114,
                      117,120,125,129,137,140,143,146,149,152,155,158,161,164,167,170,173,176,
                      179,182,185,188,191,194,197,200,271:273,276:280,283,334)]
rm(dataall)

####TRAIN/TEST SPLIT####
#split 80/20 train test
inTrain <- createDataPartition(y = datavar$TARGET, p = 0.8, list = FALSE) 
train <- datavar[ inTrain,]
test <- datavar[ -inTrain,]
rm(datavar)

#####RAW TRAINNIG SET####
#create labels for data predictions
IDSEX<- test[,c(1,5)]

#MODEL 1 RAW
model1RAW<- gbm(TARGET ~ .-SK_ID_CURR, distribution="bernoulli",
                data = train, n.trees=100, interaction.depth = 4, shrinkage = 0.1) 
pred1RAW<- predict(object=model1RAW, newdata=test, n.trees=100, type = "response")
pred1RAWdf <- data.frame(pred1RAW)
pred1RAWbinary <- as.factor(ifelse(pred1RAW>0.2,1,0)) #80% economic threshold
test$TARGET <- as.factor(test$TARGET)
CM1RAW<- confusionMatrix(pred1RAWbinary,test$TARGET);print(CM1RAW)

####ROC Curve
model1RAW_ROC_prediction <- prediction(pred1RAW, test$TARGET)
model1RAW_ROC <- performance(model1RAW_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(model1RAW_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(model1RAW_ROC_prediction,"auc") #Create AUC data
model1RAW_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
model1RAW_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
rm(model1RAW,pred1RAW, pred1RAWbinary);gc()

#MODEL 2 RAW
trainnosex <- train[-c(1,5,6)]
model2RAW<- gbm(TARGET ~.,distribution="bernoulli",
             data = trainnosex, n.trees=100, interaction.depth = 4, shrinkage = 0.1) 
pred2RAW<- predict(object=model2RAW, newdata=test, n.trees=100, type = "response")
pred2RAWdf <- data.frame(pred2RAW)
pred2RAWbinary <- as.factor(ifelse(pred2RAW>0.2,1,0)) #80% economic threshold
CM2RAW<- confusionMatrix(pred2RAWbinary,test$TARGET);print(CM2RAW)

####ROC Curve
model2RAW_ROC_prediction <- prediction(pred2RAW, test$TARGET)
model2RAW_ROC <- performance(model2RAW_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(model2RAW_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(model2RAW_ROC_prediction,"auc") #Create AUC data
model2RAW_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
model2RAW_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
rm(model2RAW,pred2RAW, pred2RAWbinary);gc()

####MINORITY TRAINING SET####
#undersampling procedure
trainmale <- train[train$`CODE_GENDER=M` == 1, ]
trainfemale <- train[train$`CODE_GENDER=F` == 1, ]
minTrain <- createDataPartition(y = trainfemale$TARGET, 
                                p = ((round((nrow(trainmale))/0.8))-nrow(trainmale))/nrow(trainfemale), list = FALSE) 
trainMINfemale <- trainfemale[ minTrain,]
trainMIN <- rbind.data.frame(trainmale,trainMINfemale)
rm(trainfemale)

#remove variables with no variance (additional variables b/c of smaller dataset)
trainMIN <- trainMIN[-c(32,129,130,222)]

#MODEL 1 MINORITY
model1MIN<- gbm(TARGET ~ .-SK_ID_CURR,distribution="bernoulli",
             data = trainMIN, n.trees=100, interaction.depth = 4, shrinkage = 0.1) 
pred1MIN<- predict(object=model1MIN, newdata=test, n.trees=100, type = "response")
pred1MINdf <- data.frame(pred1MIN)
pred1MINbinary <- as.factor(ifelse(pred1MIN>0.2,1,0)) #80% economic threshold
CM1MIN <- confusionMatrix(pred1MINbinary,test$TARGET);print(CM1MIN)

####ROC Curve
model1MIN_ROC_prediction <- prediction(pred1MIN, test$TARGET)
model1MIN_ROC <- performance(model1MIN_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(model1MIN_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(model1MIN_ROC_prediction,"auc") #Create AUC data
model1MIN_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
model1MIN_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
rm(model1MIN,pred1MIN, pred1MINbinary);gc()

#MODEL 2 MINORITY
trainnosexMIN <- trainMIN[-c(1,5,6)]
model2MIN<- gbm(TARGET ~ .,distribution="bernoulli",
             data = trainnosexMIN, n.trees=100, interaction.depth = 4, shrinkage = 0.1) 
pred2MIN<- predict(object=model2MIN, newdata=test, n.trees=100, type = "response")
pred2MINdf <- data.frame(pred2MIN)
pred2MINbinary <- as.factor(ifelse(pred2MIN>0.2,1,0)) #80% economic threshold
CM2MIN <- confusionMatrix(pred2MINbinary,test$TARGET);print(CM2MIN)

####ROC Curve
model2MIN_ROC_prediction <- prediction(pred2MIN, test$TARGET)
model2MIN_ROC <- performance(model2MIN_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(model2MIN_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(model2MIN_ROC_prediction,"auc") #Create AUC data
model2MIN_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
model2MIN_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
rm(model2MIN,pred2MIN, pred2MINbinary);gc()

####REBALANCED TRAINING SET####
#rebalancing procedure
in50Train <- createDataPartition(y = trainmale$TARGET, 
                                p = (nrow(trainMINfemale)/nrow(trainmale)), list = FALSE) 
train50male <- trainmale[ in50Train,]
train50 <- rbind.data.frame(train50male,trainMINfemale)
rm(trainmale,trainMINfemale,train50male)

#remove variables with no variance (additional variables b/c of smaller dataset)
train50 <- train50[-c(32,125,126,129,130,222)]

#MODEL 1 REBALANCED
model150<- gbm(TARGET ~ .-SK_ID_CURR,distribution="bernoulli",
                data = train50, n.trees=100, interaction.depth = 4, shrinkage = 0.1) 
pred150<- predict(object=model150, newdata=test, n.trees=100, type = "response")
pred150df <- data.frame(pred150)
pred150binary <- as.factor(ifelse(pred150>0.2,1,0)) #80% economic threshold
CM150 <- confusionMatrix(pred150binary,test$TARGET);print(CM150)

####ROC Curve
model150_ROC_prediction <- prediction(pred150, test$TARGET)
model150_ROC <- performance(model150_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(model150_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(model150_ROC_prediction,"auc") #Create AUC data
model150_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
model150_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
rm(model150,pred150, pred150binary);gc()

#MODEL 2 REBALANCED
trainnosex50 <- train50[-c(1,5,6)]
model250<- gbm(TARGET ~ .,distribution="bernoulli",
                data = trainnosex50, n.trees=100, interaction.depth = 4, shrinkage = 0.1) 
pred250<- predict(object=model250, newdata=test, n.trees=100, type = "response")
pred250df <- data.frame(pred250)
pred250binary <- as.factor(ifelse(pred250>0.2,1,0)) #80% economic threshold
CM250 <- confusionMatrix(pred250binary,test$TARGET);print(CM250)

####ROC Curve
model250_ROC_prediction <- prediction(pred250, test$TARGET)
model250_ROC <- performance(model250_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(model250_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(model250_ROC_prediction,"auc") #Create AUC data
model250_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
model250_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
rm(model250,pred250, pred250binary);gc()


####GLM MODEL COMPARISON####
dataglm[c(2)] <- lapply(dataglm[c(2)],as.factor)

#create model matrix
MM <- model_matrix(dataglm, TARGET ~. -1)
dataMM <- merge(MM, IDTARGET, by="SK_ID_CURR")

####GLM TRAIN/TEST SPLIT ####
#split 80/20 train test
inTrain <- createDataPartition(y = dataMM$TARGET, p = 0.8, list = FALSE)
trainglm <- dataMM[ inTrain,]
testglm <- dataMM[ -inTrain,]
rm(inTrain)

#create labels for data predictions
IDSEXglm<- testglm[,c(1,4)]

####GLM MINORITY TRAINING SET####
#undersampling procedure
trainglmmale <- trainglm[trainglm$CODE_GENDERM == 1, ]
trainglmfemale <- trainglm[trainglm$CODE_GENDERM == 0, ]
minTrain <- createDataPartition(y = trainglmfemale$TARGET, 
                                p = ((round((nrow(trainglmmale))/0.8))-nrow(trainglmmale))/nrow(trainglmfemale), list = FALSE) 
trainglmMINfemale <- trainglmfemale[ minTrain,]
trainglmMIN <- rbind.data.frame(trainglmmale,trainglmMINfemale)
rm(trainglmfemale, minTrain)

#MODEL 1 MINORITY (GLM)
glmmodel1MIN<- glm(TARGET~.-SK_ID_CURR,
                   data = trainglmMIN, family=binomial, na.action=na.exclude) 
summary(glmmodel1MIN)
predglm1MIN<- predict(glmmodel1MIN, newdata= testglm, type = "response")
predglm1MINdf <- data.frame(predglm1MIN)
predglm1MINbinary <- as.factor(ifelse(predglm1MIN>0.2,1,0)) #80% economic threshold
testglm$TARGET <- as.factor(testglm$TARGET)
CMglm1MIN<- confusionMatrix(predglm1MINbinary,testglm$TARGET);print(CMglm1MIN)

####ROC Curve
glmmodel1MIN_ROC_prediction <- prediction(predglm1MIN, testglm$TARGET)
glmmodel1MIN_ROC <- performance(glmmodel1MIN_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(glmmodel1MIN_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(glmmodel1MIN_ROC_prediction,"auc") #Create AUC data
glmmodel1MIN_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
glmmodel1MIN_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
rm(glmmodel1MIN,predglm1MIN, predglm1MINbinary);gc()

#MODEL 2 MINORITY (GLM)
trainnosexglmMIN <- trainglmMIN[-c(1,4,5)]
glmmodel2MIN<- glm(TARGET ~.,data = trainnosexglmMIN, 
                   family=binomial, na.action=na.exclude) 
summary(glmmodel2MIN)
predglm2MIN<- predict(glmmodel2MIN, newdata= testglm, type = "response")
predglm2MINdf <- data.frame(predglm2MIN)
predglm2MINbinary <- as.factor(ifelse(predglm2MIN>0.2,1,0)) #80% economic threshold
testglm$TARGET <- as.factor(testglm$TARGET)
CMglm2MIN<- confusionMatrix(predglm2MINbinary,testglm$TARGET);print(CMglm2MIN)

####ROC Curve
glmmodel2MIN_ROC_prediction <- prediction(predglm2MIN, testglm$TARGET)
glmmodel2MIN_ROC <- performance(glmmodel2MIN_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(glmmodel2MIN_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(glmmodel2MIN_ROC_prediction,"auc") #Create AUC data
glmmodel2MIN_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
glmmodel2MIN_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
rm(glmmodel2MIN,predglm2MIN, predglm2MINbinary);gc()


####GLM REBALANCED TRAINING SET####
##rebalancing procedure
#rebalancing procedure
in50Train <- createDataPartition(y = trainglmmale$TARGET, 
                                 p = (nrow(trainglmMINfemale)/nrow(trainglmmale)), list = FALSE) 
trainglm50male <- trainglmmale[ in50Train,]
trainglm50 <- rbind.data.frame(trainglm50male,trainglmMINfemale)
rm(in50Train, trainglmmale,trainglmMINfemale,trainglm50male)

#MODEL 1 REBALANCED (GLM)
glmmodel150<- glm(TARGET ~.-SK_ID_CURR,
                  data = trainglm50, family=binomial, na.action=na.exclude) 
summary(glmmodel150)
predglm150<- predict(glmmodel150, newdata= testglm, type = "response")
predglm150df <- data.frame(predglm150)
predglm150binary <- as.factor(ifelse(predglm150>0.2,1,0)) #80% economic threshold
testglm$TARGET <- as.factor(testglm$TARGET)
CMglm150<- confusionMatrix(predglm150binary,testglm$TARGET);print(CMglm150)

####ROC Curve
glmmodel150_ROC_prediction <- prediction(predglm150, testglm$TARGET)
glmmodel150_ROC <- performance(glmmodel150_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(glmmodel150_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(glmmodel150_ROC_prediction,"auc") #Create AUC data
glmmodel150_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
glmmodel150_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
rm(glmmodel150,predglm150, predglm150binary);gc()

#MODEL 2 REBALANCED (GLM)
trainnosexglm50 <- trainglm50[-c(1,4,5)]
glmmodel250<- glm(TARGET ~., data = trainnosexglm50,
                  family=binomial, na.action=na.exclude) 
summary(glmmodel250)
predglm250<- predict(glmmodel250, newdata= testglm, type = "response")
predglm250df <- data.frame(predglm250)
predglm250binary <- as.factor(ifelse(predglm250>0.2,1,0)) #80% economic threshold
testglm$TARGET <- as.factor(testglm$TARGET)
CMglm250<- confusionMatrix(predglm250binary,testglm$TARGET);print(CMglm250)

####ROC Curve
glmmodel250_ROC_prediction <- prediction(predglm250, testglm$TARGET)
glmmodel250_ROC <- performance(glmmodel250_ROC_prediction,"tpr","fpr") #Create ROC curve data
plot(glmmodel250_ROC) #Plot ROC curve

####AUC (area under curve)
auc.tmp <- performance(glmmodel250_ROC_prediction,"auc") #Create AUC data
glmmodel250_AUC <- as.numeric(auc.tmp@y.values) #Calculate AUC
glmmodel250_AUC #Display AUC value: 90+% - excellent, 80-90% - very good, 70-80% - good, 60-70% - so so, below 60% - not much value
rm(glmmodel250,predglm250, predglm250binary);gc()


#####print to csv#####
write.csv(IDSEX,  "IDSEX.csv")
write.csv(pred1RAWdf, "pred1RAWdf.csv")
write.csv(pred2RAWdf, "pred2RAWdf.csv")
write.csv(pred1MINdf, "pred1MINdf.csv")
write.csv(pred2MINdf, "pred2MINdf.csv")
write.csv(pred150df, "pred150df.csv")
write.csv(pred250df,"pred250df.csv")
write.csv(IDSEXglm, "IDSEXglm.csv")
write.csv(predglm1MINdf, "predglm1MINdf.csv")
write.csv(predglm2MINdf, "predglm2MINdf.csv")
write.csv(predglm150df, "predglm150df.csv")
write.csv(predglm250df, "predglm250df.csv")
