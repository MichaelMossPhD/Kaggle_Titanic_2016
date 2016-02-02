library(gmodels)
library(ggplot2)      # supports enhanced graphics
library(Hmisc)        # impute missing values
library(car)          # produce VIF statistics
library(coefplot)     # coefficient plot
library(plyr)         # for the revalue function 
library(bestglm)      # cross-validation for logistic regression
library(caret)        # cross-validation for logistic regression
library(kernlab)      # assist with SVM
library(rpart)        # classification and refression trees
library(partykit)     # treeplots
library(randomForest) # random forests
library(rattle)       # support tree graphics
library(rpart.plot)   # support tree graphics
library(RColorBrewer) # support tree graphics
library(ada)          # Support boosting model
library(party)        # supports conditional inference tree model
library(splines)

# --------------------------------------------------------------------------

## Fetch data from GitHub ##

# Read Data function
readData <- function(path.name, file.name, column.types, missing.types) {
  read.csv( url( paste(path.name, file.name, sep="") ), colClasses=column.types, na.strings=missing.types )
}

# Arguments that are passed to the readData function
Titanic.path <- "https://raw.github.com/MichaelMossPhD/Kaggle_Titanic_2016/master/"

train.data <- "train.csv"
test.data <- "test.csv"

train.column.types <- c('integer',   # PassengerId
                        'factor',    # Survived 
                        'factor',    # Pclass
                        'character', # Name
                        'factor',    # Sex
                        'numeric',   # Age
                        'integer',   # SibSp
                        'integer',   # Parch
                        'character', # Ticket
                        'numeric',   # Fare
                        'character', # Cabin
                        'factor'     # Embarked
)

test.column.types <- train.column.types[-2]     # remove Survived column

missing.types <- c("NA", "")

# Get the training and test data files
train <- readData(Titanic.path, train.data, train.column.types, missing.types)
test <- readData(Titanic.path, test.data, test.column.types, missing.types)

# Verify objects as dataframes
class(train)
class(test)

# Display objects' structure
str(train)
str(test)

# Look at the first rows of the train object
head(train)

# --------------------------------------------------------------------------

## Descriptive Statistics ##

# Summary statistics for the train object
summary(train)

# Cross-tabulation of survived by sex. Survived: 0 = N, 1 = Yes; Sex: male, female
CrossTable(x = train$Survived, y = train$Sex, chisq = TRUE) # Factors

# Cross-tabulation of survived by travelling class. Survived: 0 = N, 1 = Yes; Pclass: 1 = 1st, 2 = 2nd, 3 = 3rd
CrossTable(x = train$Survived, y = train$Pclass, chisq = TRUE) # factors

# Cross-tabulation of survived by embarked. Survived: 0 = N, 1 = Yes; Embarked: C = Cherbourg, Q = Queenstown, S = Southampton
CrossTable(x = train$Survived, y = train$Embarked, chisq = TRUE) # factors

## Data Visualization ##

# Barplots

# Survived
barplot(table(train$Survived), names.arg = c("Perished", "Survived"), main = "Passenger Fate", col = c("red","green"))
# Pclass
barplot(table(train$Pclass), names.arg = c("First Class", "Second Class", "Third Class"), main = "Passengers by Travelling Class",
        col = c("green","yellow", "red" ))
# Sex
barplot(table(train$Sex), names.arg = c("Female", "Male"), main = "Passenger Sex", col = c("pink","blue"))
# Siblings/Spouses
barplot(table(train$SibSp), main = "Number of Siblings/Spouses Abroad")
# Parents/children
barplot(table(train$Parch), main = "Number of Parents/Children Abroad")
# Embarked
barplot(table(train$Embarked), names.arg = c("Cherbourg", "Queenstown", "Southhampton"), main = "Port of Embarkation")

# Histograms

# Survived
ggplot(train, aes(x = Survived)) + geom_histogram(binwidth = 5, fill = "white", colour = "black")
# Pclass
ggplot(train, aes(x = Pclass)) + geom_histogram(binwidth = 5, fill = "white", colour = "black")
# Sex
ggplot(train, aes(x = Sex)) + geom_histogram(binwidth = 5, fill = "white", colour = "black")
# Age
ggplot(train, aes(x = Age)) + geom_histogram(binwidth = 5, fill = "white", colour = "black")
# Siblings/Spouses
ggplot(train, aes(x = SibSp)) + geom_histogram(binwidth = 1, fill = "white", colour = "black")
# Parents/children
ggplot(train, aes(x = Parch)) + geom_histogram(binwidth = 1, fill = "white", colour = "black")
# Embarked
ggplot(train, aes(x = Embarked)) + geom_histogram(binwidth = 1, fill = "white", colour = "black")

# Density curves

# Age
ggplot(train, aes(x = Age)) + geom_line(stat = "density")
# Siblings/Spouses
ggplot(train, aes(x = SibSp)) + geom_line(stat = "density")
# Parents/children
ggplot(train, aes(x = Parch)) + geom_line(stat = "density")

# Histogram with density curve

# Age
ggplot(train, aes(x = Age, y = ..density..)) + geom_histogram(fill = "cornsilk", colour = "grey60", size = .2) + geom_density() + xlim(35,105)

# Sorted by Index

# Age
plot(sort(train$Age), ylab = "Sorted Age")

# Bivariate Plots #

# Survival rate by age
ggplot(train, aes(x = Survived, y = Age)) + geom_boxplot()

# Passenger traveling class by age.
ggplot(train, aes(x = Pclass, y = Age)) + geom_boxplot()

# Survival rate by sex
sex_survival <- table(train$Survived, train$Sex)
barplot(sex_survival, xlab = "Gender", ylab = "Number of People", main = "Passenger Fate by Sex", col = c("red","green"))
sex_survival[2] / (sex_survival[1] + sex_survival[2])
sex_survival[4] / (sex_survival[3] + sex_survival[4])

# Survival rate by passenger class
Pclass_survival <- table(train$Survived, train$Pclass)
barplot(Pclass_survival, xlab = "Cabin Class", ylab = "Number of People", main = "Passenger Fate by Traveling Class",
        col = c("red","green"))

# Percantages of survival by passenger class.
Pclass_survival

Pclass_survival[2] / (Pclass_survival[1] + Pclass_survival[2]) # First class
Pclass_survival[4] / (Pclass_survival[3] + Pclass_survival[4]) # Second class
Pclass_survival[6] / (Pclass_survival[5] + Pclass_survival[6]) # Third class

# --------------------------------------------------------------------------

## Data Preparation (Train and Test data sets) ##

# Function for imputing missing values, using the median of existing values
imputeMedian <- function(impute.var, filter.var, var.levels) {
  for (v in var.levels) {
    impute.var[ which( filter.var == v)] <- impute(impute.var[ 
      which( filter.var == v)])
  }
  return (impute.var)
}

# View how sex factor is coded
contrasts(train$Sex) # female: 0, male: 1

# Remove unnecessary variables: Ticket and Cabin
train.prep = train[-c(9,11)]
test.prep = test[-c(8,10)]

# Account for missing ages

# First, group people by their titles

# Training data
master_vector = grep("Master.", train.prep$Name, fixed = TRUE)
miss_vector = grep("Miss.", train.prep$Name, fixed = TRUE)
mrs_vector = grep("Mrs.", train.prep$Name, fixed = TRUE)
mr_vector = grep("Mr.", train.prep$Name, fixed = TRUE)
dr_vector = grep("Dr.", train.prep$Name, fixed = TRUE)
rev_vector = grep("Rev.", train.prep$Name, fixed = TRUE)
capt_vector = grep("Capt.", train.prep$Name, fixed = TRUE)
major_vector = grep("Major.", train.prep$Name, fixed = TRUE)
col_vector = grep("Col.", train.prep$Name, fixed = TRUE)
don_vector = grep("Don.", train.prep$Name, fixed = TRUE)
sir_vector = grep("Sir.", train.prep$Name, fixed = TRUE)
jonkheer_vector = grep("Jonkheer.", train.prep$Name, fixed = TRUE)
lady_vector = grep("Lady.", train.prep$Name, fixed = TRUE)
mme_vector = grep("Mme.", train.prep$Name, fixed = TRUE)
mlle_vector = grep("Mlle.", train.prep$Name, fixed = TRUE)
countess_vector = grep("Countess.", train.prep$Name, fixed = TRUE)
ms_vector = grep("Ms.", train.prep$Name, fixed = TRUE)

# Test data
test_master_vector = grep("Master.", test.prep$Name, fixed = TRUE)
test_miss_vector = grep("Miss.", test.prep$Name, fixed = TRUE)
test_ms_vector = grep("Ms.", test.prep$Name, fixed = TRUE)
test_mrs_vector = grep("Mrs.", test.prep$Name, fixed = TRUE)
test_mr_vector = grep("Mr.", test.prep$Name, fixed = TRUE)
test_dr_vector = grep("Dr.", test.prep$Name, fixed = TRUE)
test_rev_vector = grep("Rev.", test.prep$Name, fixed = TRUE)
test_capt_vector = grep("Capt.", test.prep$Name, fixed = TRUE)
test_major_vector = grep("Major.", test.prep$Name, fixed = TRUE)
test_col_vector = grep("Col.", test.prep$Name, fixed = TRUE)
test_don_vector = grep("Don.", test.prep$Name, fixed = TRUE)
test_sir_vector = grep("Sir.", test.prep$Name, fixed = TRUE)
test_jonkheer_vector = grep("Jonkheer.", test.prep$Name, fixed = TRUE)
test_lady_vector = grep("Lady.", test.prep$Name, fixed = TRUE)
test_mme_vector = grep("Mme.", test.prep$Name, fixed = TRUE)
test_mlle_vector = grep("Mlle.", test.prep$Name, fixed = TRUE)
test_countess_vector = grep("Countess.", test.prep$Name, fixed = TRUE)
test_dona_vector = grep("Dona.", test.prep$Name, fixed = TRUE)

# Second, rename every person by his/her title only

# Training data
for (i in master_vector) {
  train.prep$Name[i] = "Master"
}
for (i in miss_vector) {
  train.prep$Name[i] = "Miss"
}
for (i in mrs_vector) {
  train.prep$Name[i] = "Mrs"
}
for (i in mr_vector) {
  train.prep$Name[i] = "Mr"
}
for (i in dr_vector) {
  train.prep$Name[i] = "Dr"
}
for (i in rev_vector) {
  train.prep$Name[i] = "Rev"
}
for (i in capt_vector) {
  train.prep$Name[i] = "Sir"
}
for (i in major_vector) {
  train.prep$Name[i] = "Sir"
}
for (i in col_vector) {
  train.prep$Name[i] = "Sir"
}
for (i in sir_vector) {
  train.prep$Name[i] = "Sir"
}
for (i in don_vector) {
  train.prep$Name[i] = "Sir"
}
for (i in jonkheer_vector) {
  train.prep$Name[i] = "Sir"
}
for (i in lady_vector) {
  train.prep$Name[i] = "Lady"
}
for (i in mme_vector) {
  train.prep$Name[i] = "Lady"
}
for (i in mlle_vector) {
  train.prep$Name[i] = "Lady"
}
for (i in countess_vector) {
  train.prep$Name[i] = "Lady"
}
for (i in ms_vector) {
  train.prep$Name[i] = "Miss"
}

# Test data
for (i in test_master_vector) {
  test.prep[i, 3] = "Master"
}
for (i in test_miss_vector) {
  test.prep[i, 3] = "Miss"
}
for (i in test_ms_vector) {
  test.prep[i, 3] = "Miss"
}
for (i in test_mrs_vector) {
  test.prep[i, 3] = "Mrs"
}
for (i in test_mr_vector) {
  test.prep[i, 3] = "Mr"
}
for (i in test_dr_vector) {
  test.prep[i, 3] = "Dr"
}
for (i in test_rev_vector) {
  test.prep[i, 3] = "Rev"
}
for (i in test_capt_vector) {
  test.prep[i, 3] = "Sir"
}
for (i in test_major_vector) {
  test.prep[i, 3] = "Sir"
}
for (i in test_col_vector) {
  test.prep[i, 3] = "Sir"
}
for (i in test_don_vector) {
  test.prep[i, 3] = "Sir"
}
for (i in test_sir_vector) {
  test.prep[i, 3] = "Sir"
}
for (i in test_jonkheer_vector) {
  test.prep[i, 3] = "Sir"
}
for (i in test_lady_vector) {
  test.prep[i, 3] = "Lady"
}
for (i in test_mme_vector) {
  test.prep[i, 3] = "Lady"
}
for (i in test_mlle_vector) {
  test.prep[i, 3] = "Lady"
}
for (i in test_countess_vector) {
  test.prep[i, 3] = "Lady"
}
for (i in test_dona_vector) {
  test.prep[i, 3] = "Lady"
}

# Third, find the mean age values for each group with NAs

# Training data
master_age = round(mean(train.prep$Age[train.prep$Name == "Master"], na.rm = TRUE), digits = 2)
miss_age = round(mean(train.prep$Age[train.prep$Name == "Miss"], na.rm = TRUE), digits = 2)
mrs_age = round(mean(train.prep$Age[train.prep$Name == "Mrs"], na.rm = TRUE), digits = 2)
mr_age = round(mean(train.prep$Age[train.prep$Name == "Mr"], na.rm = TRUE), digits = 2)
dr_age = round(mean(train.prep$Age[train.prep$Name == "Dr"], na.rm = TRUE), digits = 2)

# Test data
test_master_age = round(mean(test.prep$Age[test.prep$Name == "Master"], na.rm = TRUE), digits = 2)
test_miss_age = round(mean(test.prep$Age[test.prep$Name == "Miss"], na.rm = TRUE), digits = 2)
test_mrs_age = round(mean(test.prep$Age[test.prep$Name == "Mrs"], na.rm = TRUE), digits = 2)
test_mr_age = round(mean(test.prep$Age[test.prep$Name == "Mr"], na.rm = TRUE), digits = 2)
test_dr_age = round(mean(test.prep$Age[test.prep$Name == "Dr"], na.rm = TRUE), digits = 2)

# Fourth, replace the NAs with the mean age for each group

# Training data
for (i in 1:nrow(train.prep)) {
  if (is.na(train.prep[i, 6])) {
    if (train.prep$Name[i] == "Master") {
      train.prep$Age[i] = master_age
    } else if (train.prep$Name[i] == "Miss") {
      train.prep$Age[i] = miss_age
    } else if (train.prep$Name[i] == "Mrs") {
      train.prep$Age[i] = mrs_age
    } else if (train.prep$Name[i] == "Mr") {
      train.prep$Age[i] = mr_age
    } else if (train.prep$Name[i] == "Dr") {
      train.prep$Age[i] = dr_age
    } else {
      print("Uncaught Title")
    }
  }
}

# Test data
for (i in 1:nrow(test.prep)) {
  if (is.na(test.prep[i, 5])) {
    if (test.prep[i, 3] == "Master") {
      test.prep[i, 5] = test_master_age
    } else if (test.prep[i, 3] == "Miss") {
      test.prep[i, 5] = test_miss_age
    } else if (test.prep[i, 3] == "Mrs") {
      test.prep[i, 5] = test_mrs_age
    } else if (test.prep[i, 3] == "Mr") {
      test.prep[i, 5] = test_mr_age
    } else if (test.prep[i, 3] == "Dr") {
      test.prep[i, 5] = test_dr_age
    } else {
      print(paste("Uncaught title at: ", i, sep = ""))
      print(paste("The title unrecognized was: ", test.prep[i, 3], sep = ""))
    }
  }
}

# Replace NAs in train.prep$Embarked with S; test Data is OK.
train.prep$Embarked[which(is.na(train.prep$Embarked))] <- 'S'

# Impute missings values for Fare with median fare by Pclass

# Training data
train.prep$Fare[which( train.prep$Fare == 0 )] <- NA
train.prep$Fare <- imputeMedian(train.prep$Fare, train.prep$Pclass, as.numeric(levels(train.prep$Pclass)))

# Test data
test.prep$Fare[which( test.prep$Fare == 0 )] <- NA
test.prep$Fare <- imputeMedian(test.prep$Fare, test.prep$Pclass, as.numeric(levels(test.prep$Pclass)))

## Create new variables ##

# Change dependent variable class levels to valid R variable names.
# Will change back to original values before processing predictions.
train.prep$Fate <- train.prep$Survived
train.prep$Fate <- revalue(train.prep$Fate, c("1" = "Survived", "0" = "Perished"))

# Dummy variable for child. A child is someone less than or equal to 14 years old, Child: 1

# Train data
train.prep["Child"] = NA
for (i in 1:nrow(train.prep)) { 
  if (train.prep$Age[i] <= 14) {
    train.prep$Child[i] = 1
  } else {
    train.prep$Child[i] = 0
  }
}

# Cross-tabulation of children who survived
CrossTable(x = train.prep$Fate, y = train.prep$Child, chisq = TRUE)

# Test data
test.prep["Child"] = NA
for (i in 1:nrow(test.prep)) {
  if (test.prep[i, 5] <= 14) {
    test.prep[i, 10] = 1
  } else {
    test.prep[i, 10] = 0
  }
}

# Dummy variable for mother. A mother's title is "Mrs." and has a Parent/Child value > 0, Mother: 1

# Training data
train.prep["Mother"] = NA 
for(i in 1:nrow(train.prep)) {
  if(train.prep$Name[i] == "Mrs" & train.prep$Parch[i] > 0) {
    train.prep$Mother[i] = 1
  } else {
    train.prep$Mother[i] = 0
  }
}

# Cross-tabulation of mothers who survived
CrossTable(x = train.prep$Survived, y = train.prep$Mother, chisq = TRUE)

# Test data
test.prep["Mother"] = NA
for(i in 1:nrow(test.prep)) {
  if(test.prep[i, 3] == "Mrs" & test.prep[i, 7] > 0) {
    test.prep[i, 11] = 1
  } else {
    test.prep[i, 11] = 0
  }
}

# Family size variable

# Train data
train.prep["Family"] = NA
for(i in 1:nrow(train.prep)) {
  x = train.prep$SibSp[i]
  y = train.prep$Parch[i]
  train.prep$Family[i] = x + y + 1 # Add 1 for the observation itself
}

# Test data
test.prep["Family"] = NA
for(i in 1:nrow(test.prep)) {
  test.prep[i, 12] = test.prep[i, 6] + test.prep[i, 7] + 1 # Add 1 for the observation itself
}

# FareAdj provides the fare per person, accounting for families making bulk purchases.
train.prep$FareAdj <- train.prep$Fare/(train.prep$Family)
test.prep$FareAdj <- test.prep$Fare/(test.prep$Family)

# --------------------------------------------------------------------------

## Model Development and Evaluation ##

# -----

# Logistic Regression Model - Kaggle Score = .77033 #

titantic.glm.fit <- glm(Survived ~ Age + SibSp + I(Name == "Master") + Sex*Pclass + FareAdj*Pclass, family=binomial("logit"), data=train.prep)

summary(titantic.glm.fit)  # inpsect the coefficients and their p-values
confint(titantic.glm.fit)  # examine the 95% confidence intervals
vif(titantic.glm.fit)      # Check for multicollinearity, GVIF > 5

# Build a confusion matrix

train.prep$probs = predict(titantic.glm.fit, type="response")
train.prep$probs[1:5]

train.prep$predict = rep(0, 891)
train.prep$predict[train.prep$probs > 0.5] = 1

table(train.prep$predict, train.prep$Survived)

mean(train.prep$predict==train.prep$Survived)  # Percentage of observations predicted correctly

# Measure how well the model fits (significance of the overall model
# Test whether the model with predictors fits significantly better than a model with just an intercept

# Differeince in deviance
with(titantic.glm.fit, null.deviance - deviance) # chi-square
# Difference in degrees of freedom
with(titantic.glm.fit, df.null - df.residual)    # degrees of freedom
# Get the model's p-value
with(titantic.glm.fit, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE)) # p-value

# Analysis of Deviance Table
anova(titantic.glm.fit, test = "Chisq")

# Likelihood ratio test
logLik(titantic.glm.fit)

# Coefficient plot
coefplot(titantic.glm.fit)

# Confident intervals using standard errors
confint.default(titantic.glm.fit)

# Odds ratios and 95% CI.
exp(cbind(OR = coef(titantic.glm.fit), confint(titantic.glm.fit)))

## Submit Kaggle Predictions ##

# Use the logistic regression model to generate predictions.
p.survived <- predict.glm(titantic.glm.fit, newdata = test.prep, type = "response")
# If survived predicition greater than 0.5, person lived (1), else died (0).
Survived = ifelse(p.survived > 0.5, 1, 0)

# Move predictions(Survived) into the predictions data frame
predictions <- as.data.frame(Survived)
# Add PassengerId column from test file to the predictions data frame
predictions$PassengerId <- test$PassengerId
# Write predictions to csv file for submission to Kaggle
write.csv(predictions[,c("PassengerId", "Survived")], 
          file="Titanic_Predictions.csv", row.names = FALSE, quote = FALSE)

# -----

# Logistic Regression model with cross-validation and class compression - Kaggle Score = .78469 #

# The trainControl() function creates a set of configuration options known as the control object and is used
# with the train() function

crossVal.ctrl <- trainControl(method = "repeatedcv", repeats = 3, summaryFunction = twoClassSummary, classProbs = TRUE)

# Model using train() function
set.seed(456)

titantic.glm.cv <- train(Fate ~ Age + I(Name == "Master") + I(Embarked=="S") + Sex*Pclass + FareAdj*Pclass + Family,
                         data = train.prep, method = "glm", metric = "ROC", trControl = crossVal.ctrl)

summary(titantic.glm.cv)
titantic.glm.cv

### SUBMIT PREDICTIONS ###

# Use the logistic regression model with cross-validation and class compression to generate predictions
Survived <- predict(titantic.glm.cv, newdata = test.prep)
# Change class back to original format
Survived <- revalue(Survived, c("Survived" = 1, "Perished" = 0))

# Move predictions(Survived) into the predictions data frame
predictions <- as.data.frame(Survived)
# Add PassengerId column from test file to the predictions data frame
predictions$PassengerId <- test$PassengerId
# Write predictions to csv file for submission to Kaggle
write.csv(predictions[,c("PassengerId", "Survived")], 
          file="Titanic_Predictions.csv", row.names = FALSE, quote = FALSE)

# -----

# Service Vector Machine - Kaggle Score = .78947 #

set.seed(456)

titantic.svm <- train(Fate ~ Age + Family + Sex + Pclass + FareAdj + Embarked,
                      data = train.prep,  method = "svmRadial", tuneLength = 9, preProcess = c("center", "scale"),
                      metric = "ROC", trControl = crossVal.ctrl)
titantic.svm

# Use the Service Vector Machine to generate predictions
Survived <- predict(titantic.svm, newdata = test.prep)
# Change class back to original format
Survived <- revalue(Survived, c("Survived" = 1, "Perished" = 0))

# Move predictions(Survived) into the predictions data frame
predictions <- as.data.frame(Survived)
# Add PassengerId column from test file to the predictions data frame
predictions$PassengerId <- test$PassengerId
# Write predictions to csv file for submission to Kaggle
write.csv(predictions[,c("PassengerId", "Survived")], 
          file="Titanic_Predictions.csv", row.names = FALSE, quote = FALSE)

# -----

# Regression Tree - Kaggle Score = .78469 #

titantic.rt = rpart(Survived ~ Age + Family + Sex + Pclass + FareAdj + Embarked, data = train.prep,
                    method = "class")

fancyRpartPlot(titantic.rt)   # Display the tree

print(titantic.rt$cptable)    # Determine the optimal number of splits in the tree
plotcp(titantic.rt)

# Use the Regression Tree to generate predictions
Survived <- predict(titantic.rt, newdata = test.prep, type = "class")

# Move predictions(Survived) into the predictions data frame
predictions <- as.data.frame(Survived)
# Add PassengerId column from test file to the predictions data frame
predictions$PassengerId <- test$PassengerId
# Write predictions to csv file for submission to Kaggle
write.csv(predictions[,c("PassengerId", "Survived")], 
          file="Titanic_Predictions.csv", row.names = FALSE, quote = FALSE)

# -----

# Random Forrest Regression - Kaggle Score = .78469 #

# Create a data frame with only tuning parameter for this model.
rf.grid <- expand.grid(.mtry = c(2, 3, 4))

set.seed(456)

titantic.rf <- train(Fate ~ Child + Family + Embarked + FareAdj + Sex*Pclass, data = train.prep,
                            method = "rf", metric = "ROC", tunegrid = rf.grid, trControl = crossVal.ctrl)
print(titantic.rf)


#Use the random forest model (optimal training version) to generate predictions.
Survived <- predict(titantic.rf, newdata = test.prep)
# Change class back to original format for models using the train() function.
Survived <- revalue(Survived, c("Survived" = 1, "Perished" = 0))

# Move predictions(Survived) into the predictions data frame.
predictions <- as.data.frame(Survived)
# Add PassengerId column from test data file to the predictions data frame.
predictions$PassengerId <- test$PassengerId
# Write predictions to csv file for submission to Kaggle
write.csv(predictions[,c("PassengerId", "Survived")], 
          file="Titanic_Predictions.csv", row.names=FALSE, quote=FALSE)

# -----

# ADA Boosting - Kaggle Score = .75598 #

# Create a data frame with these three variables as column names and one row per tuning variable combination
ada.grid <- expand.grid(.iter = c(50, 100),   # Number of boosting iterations, default=50.
                        .maxdepth = c(4, 8),  # Depth of trees.
                        .nu = c(0.1, 1))      # Shrinkage parameter, default=1.

set.seed(456)

titantic.ada <- train(Fate ~ Sex + Age + Family + Child + Pclass + Embarked + FareAdj, data = train.prep,
                      method = "ada", metric = "ROC", tuneGrid = ada.grid, trControl = crossVal.ctrl)

titantic.ada           # display the tuning parameters

plot(titantic.ada)     # display ada accuracy profile.

# Optimum model. #

# Apply configuration settings to new model
ada.optimal.grid <- expand.grid(.iter = c(100),
                                .maxdepth = c(4),
                                .nu = c(0.1))

set.seed(456)

titantic.ada.optimal <- train(Fate ~ Sex + Age + Family + Child + Pclass + Embarked + FareAdj,
                                     data = train.prep, method = "ada", metric = "ROC",
                                     tuneGrid = ada.optimal.grid, trControl = crossVal.ctrl)

titantic.ada.optimal

# Use the boosting model to generate predictions
Survived <- predict(titantic.ada.optimal, newdata = test.prep)
# Change class back to original format for models using the train() function.
Survived <- revalue(Survived, c("Survived" = 1, "Perished" = 0))

# Move predictions(Survived) into the predictions data frame.
predictions <- as.data.frame(Survived)
# Add PassengerId column from test data file to the predictions data frame.
predictions$PassengerId <- test.prep$PassengerId
# Write predictions to csv file for submission to Kaggle
write.csv(predictions[,c("PassengerId", "Survived")], 
          file="Titanic_Predictions.csv", row.names=FALSE, quote=FALSE)

# -----

# Conditional Inference Tree - Kaggle Score = .77033 #

set.seed(456)

titantic.cit <- train(Survived ~ Child + Family + FareAdj + Embarked + Sex + Pclass, data = train.prep,
                      method = "ctree", trControl = crossVal.ctrl, controls = cforest_unbiased(ntree=20000, mtry=7))

titantic.cit

# use the conditional inference tree to generate predictions
Survived <- predict(titantic.cit, test.prep, type = "raw")

# Move predictions(Survived) into the predictions data frame.
predictions <- as.data.frame(Survived)
# Add PassengerId column from test data file to the predictions data frame.
predictions$PassengerId <- test.prep$PassengerId
# Write predictions to csv file for submission to Kaggle
write.csv(predictions[,c("PassengerId", "Survived")], 
          file="Titanic_Predictions.csv", row.names=FALSE, quote=FALSE)









# --------------------------------------------------

### Visualization of Models Using the train() Function ###

crossVal.values <- resamples(list(Logit = trainModel.glm.tune, Ada = trainModel.ada.optimal.tune, RF = trainModel.rf.tune,
                                  CIT = trainModel.cit.tune))
dotplot(crossVal.values, metric = "ROC")

# --------------------------------------------------



