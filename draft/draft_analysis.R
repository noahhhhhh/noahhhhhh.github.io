# this is a draft analysis where the main purpose is to find the attribute of the data and
# get a basic idea on how to predict, via EDA

# set up a working directory, please do not hack me, please...
setwd("/Volumes/Data Science/Google Drive/learning_data_science/Coursera/practical_machine_learning/noahhhhhh.github.io/")

# load the training data set
setwd("/Volumes/Data Science/Google Drive/learning_data_science/Coursera/practical_machine_learning/noahhhhhh.github.io/data")
# download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "training_raw.csv", method = "curl")
# download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "testing_raw.csv", method = "curl")

df_training_raw <- read.csv("training_raw.csv")
df_testing_raw <- read.csv("testing_raw.csv")

# take a snap of the training raw data set
dim(df_training_raw)
str(df_training_raw)
table(df_training_raw$classe)

# partition the training raw data set into a train and validation, by .8 to .2 proportaion
library(caret)
ind_training_train <- createDataPartition(df_training_raw$classe, p = .8, list = F)
df_training_train <- df_training_raw[ind_training_train, ]
df_training_valid <- df_training_raw[-ind_training_train, ]

# preprocess the train data set
# 1. dummyVars it
dv_training_train <- dummyVars(classe ~., data = df_training_train)
mx_training_train_dv <- predict(dv_training_train, newdata = df_training_train)
df_training_train_dv <- data.frame(mx_training_train_dv)

mx_training_valid_dv <- predict(dv_training_train, newdata = df_training_valid)
df_training_valid_dv <- data.frame(mx_training_valid_dv)

# 2. neaZeroVar it
nzv_training_train_dv <- nearZeroVar(df_training_train_dv)
df_training_train_dv_nzv <- df_training_train_dv[, - nzv_training_train_dv]
df_training_valid_dv_nzv <- df_training_valid_dv[, - nzv_training_train_dv]

# 3. remove NAs
df_training_train_dv_nzv_rmna <- df_training_train_dv_nzv[complete.cases(df_training_train_dv_nzv), ]

# 3. standardize it
pre_sd_training_train_dv_nzv <- preProcess(df_training_train_dv_nzv, method = c("center", "scale"))
df_training_train_dv_nzv_sd <- predict(pre_sd_training_train_dv_nzv, df_training_train_dv_nzv)
df_training_valid_dv_nzv_sd <- predict(pre_sd_training_train_dv_nzv, df_training_valid_dv_nzv)

# 4. feature selection -- filtering by correlation >.75
mx_cor <- cor(df_training_train_dv_nzv_sd)
ind_cor <- findCorrelation(mx_cor, cutoff = .75)
df_training_train_dv_sd_nzv_filtered <- df_training_train_dv_nzv_sd[, - ind_cor]
df_training_valid_dv_sd_nzv_filtered <- df_training_valid_dv_nzv_sd[, - ind_coe]

# 5. principal component it
pre_pca_training_train_dv_nzv_sd <- preProcess(df_training_train_dv_nzv_sd, method = "pca")
df_training_train_dv_nzv_sd_pca <- predict(pre_pca_training_train_dv_nzv_sd, df_training_train_dv_nzv_sd)
df_training_valid_dv_nzv_sd_pca <- predict(pre_pca_training_train_dv_nzv_sd, df_training_valid_dv_nzv_sd)


# 5. give it a go, using decision tree
modelTree <- train(df_training_train$classe ~., method = "rpart", data = df_training_train_dv_nzv_sd_pca)

# what to do with those null value!!

