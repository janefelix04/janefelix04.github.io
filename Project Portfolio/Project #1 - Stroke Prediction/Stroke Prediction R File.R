library(caret)
library(ggplot2)

# Deleting BMI feature
Strokedata$bmi <- NULL

# one hot encode (changing text to numbers)

dmy <- dummyVars(" ~ .", data = Strokedata)
TrsfStroke <- data.frame(predict(dmy, newdata = Strokedata))

# Convert stroke variable to factor
TrsfStroke$stroke <- Strokedata$stroke
is.factor(TrsfStroke$stroke)
TrsfStroke$stroke <- as.factor(TrsfStroke$stroke)

# (dataset$feature,p= % in test set, list = true/false to store/not store results in list)
Partition <- createDataPartition(TrsfStroke$stroke, p = 0.7, list = FALSE)
trainset <- TrsfStroke[Partition, ]
validationset <- TrsfStroke[-Partition, ]

# just to visualize/test, not necessary
head(trainset)
head(validationset)
summary(trainset)

logisticmodel <- glm(stroke ~  age + hypertension + work_typeSelf.employed + avg_glucose_level 
                     + smoking_statusnever.smoked  + smoking_statussmokes, data = trainset, 
                     family = "binomial")

summary(logisticmodel)

# Creating actual prediction model
Logistic.Stroke <- predict(logisticmodel, validationset, type = "response")

# Creating new column for predictions
validationset$PredictStroke <- ifelse(Logistic.Stroke >= 0.25, 1, 0)
validationset$stroke <- as.numeric(as.character(validationset$stroke))
head(validationset)

# Determine accuracy
Accuracy <- mean(validationset$PredictStroke == validationset$stroke)
print(Accuracy)

# Set factors for confusion matrix
validationset$PredictStroke <- as.factor(validationset$PredictStroke)
validationset$stroke <- as.factor(validationset$stroke)

cm <- confusionMatrix(validationset$PredictStroke, validationset$stroke)
print(cm)