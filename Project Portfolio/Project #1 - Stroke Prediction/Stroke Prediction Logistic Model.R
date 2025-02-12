## Creating Logistic Model to Predict Strokes (& find most influential features)

#1: Importing libraries
library(caret)
library(ggplot2)

#2: Making bmi feature NULL, otherwise it messes with data
strokedataoriginal$bmi <- NULL

#3: One-hot encoding (categorical features -> numberical dummy variables)
dmy <- dummyVars(" ~ . ", data = strokedataoriginal)

#4: Applying transformation to data-set and ensuring it is stored as data frame
TrsfStrokedata <- (data.frame(predict(dmy, newdata = strokedataoriginal)))

#5: Reassigning original stroke column back to transformed set 
TrsfStrokedata$stroke <- strokedataoriginal$stroke

#6: Checking if "stroke" is a factor
is.factor(TrsfStrokedata$stroke)

#7: Changing "stroke" to a factor (we want it to be categorical)
TrsfStrokedata$stroke <- as.factor(TrsfStrokedata$stroke)

#8: Making partition for training set and test set, choosing to not store in list
Partition <- createDataPartition(TrsfStrokedata$stroke,p = 0.7, list = FALSE)

#9: Assigning training set partition
Trainset <- TrsfStrokedata[Partition,]

#10: Assigning test set partition
Testset <- TrsfStrokedata[-Partition,]

#11: Creating Logistic Model for training (leave out one of each past categorical)
LogisticTrainModel <- glm(stroke ~ age + hypertension + heart_disease + work_typeSelf.employed + work_typePrivate
                          + Residence_typeRural + avg_glucose_level + smoking_statusformerly.smoked + smoking_statussmokes
                          + smoking_statusnever.smoked + work_typeGovt_job, data = Trainset, family = "binomial")

#12: Generating summary using training logistic model - will show us helpful info such as P-values
#12.1: Use this summary to take out features from step #11 that have very little importance
summary(LogisticTrainModel)

#13: Creating predictor & having it use our trained model with the test set
Predict.Stroke <- predict(LogisticTrainModel, Testset, type = "response")

#14: Making new column in TESTSET data-set for stroke predictions with if-statement
#14.1: Due to the very small percentage of strokes in this data-set compared to no strokes,
#14.1.1: I need to set the % lower to make the model more accurate, and change it as I test to find most accurate
Testset$PredictStroke <- ifelse(Predict.Stroke >= 0.25,1,0)

#15: Ensuring that we are getting a numeric value for binary prediction of stroke
Testset$stroke <- as.numeric(as.character(Testset$stroke))

#16: Testing head to see if the model is predicting anything right
head(Testset)

#17: Creating accuracy variable (comparing predicted strokes to actual)
Accuracy <- mean(Testset$PredictStroke == Testset$stroke)

#18: Showing accuracy- keep using this to fine-tune and change parts to get accuracy up
print(Accuracy)

#19: Prepping to create a confusion matrix (for a detailed performance breakdown)- needs factors
Testset$PredictStroke <- as.factor(Testset$PredictStroke)
Testset$stroke <- as.factor(Testset$stroke)

#20: Creating confusion matrix
CM <- confusionMatrix(Testset$PredictStroke, Testset$stroke)

#21: Printing confusion matrix
print(CM)

## After running many times, accuracy averaging around 93-94%
## However, due to the limited number of positives (strokes) in the original dataset and the small size,
## there is some inconsistency each time it is run, primarily with what features the model thinks are important
## (Though the top 2-3 tend to remain consistent)
