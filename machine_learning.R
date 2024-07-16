# Load necessary libraries
library(dplyr)
library(tidyr)
library(readr)
library(car)
library(ggplot2)
library(factoextra)
library(tibble)
library(caret)
library(cluster)
library(multcomp)
library(e1071)
library(class)
library(gbm)
library(rpart)

# Read the data from CSV files
Features <- read_csv("All_features.csv")

# Convert necessary columns to factors
Features$emotion_id <- as.factor(Features$emotion_id)
Features$arousal_class <- as.factor(Features$arousal_class)
Features$valence_class <- as.factor(Features$valence_class)
Features$pn <- as.factor(Features$pn)
Features$quadrant <- as.factor(Features$quadrant)

# Ensure all the required columns exist
required_columns <- c("mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
                      "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10",
                      "mfcc_11", "mfcc_12", "mfcc_13", "spectral_centroid",
                      "spectral_bandwidth", "zero_crossing_rate", "rmse",
                      "mean_pressure", "max_pressure",
                      "pressure_variance", "pressure_gradient",
                      "iqr_force", "contact_area",
                      "rate_of_pressure_change", "pressure_std",
                      "num_touches", "max_touch_duration", "mean_duration")

# Check if all required columns are present
missing_columns <- setdiff(required_columns, colnames(Features))
if(length(missing_columns) > 0) {
  stop("The following required columns are missing: ", paste(missing_columns, collapse = ", "))
}

# Combine the specified features
combined_features <- Features %>%
  dplyr::select(all_of(required_columns))

# Scale numeric features
scaled_features <- scale(combined_features)
scaled_combined_features <- as.data.frame(scaled_features)
scaled_combined_features$emotion_id <- Features$emotion_id
scaled_combined_features$pn <- Features$pn

# Visualize the distribution of features across emotion classes
feature_names <- colnames(scaled_combined_features)[-ncol(scaled_combined_features) - 1]

for (feature in feature_names) {
  p <- ggplot(scaled_combined_features, aes_string(x = "emotion_id", y = feature, fill = "emotion_id")) +
    geom_boxplot() +
    theme_minimal() +
    labs(title = paste("Distribution of", feature, "Across Emotion Classes"), x = "Emotion Class", y = feature)
  print(p)
}

# Prepare data for classification
set.seed(100)

# Split data by "pn" to ensure training and test datasets are from different "pn"
unique_pn <- unique(scaled_combined_features$pn)
num_train_pn <- round(length(unique_pn) * 0.8)
train_pn <- sample(unique_pn, num_train_pn)
trainData <- scaled_combined_features %>% filter(pn %in% train_pn)
testData <- scaled_combined_features %>% filter(!pn %in% train_pn)

# Remove the 'pn' column from training and test data using base R
trainData <- trainData[ , !(names(trainData) %in% c("pn"))]
testData <- testData[ , !(names(testData) %in% c("pn"))]

# Train a Random Forest classifier with 10-fold cross-validation
rf_control <- trainControl(method = "cv", number = 10)
rf_model <- train(emotion_id ~ ., data = trainData, method = "rf", trControl = rf_control)

# Predict on test data
rf_predictions <- predict(rf_model, newdata = testData)

# Confusion matrix
rf_cm <- confusionMatrix(rf_predictions, testData$emotion_id)
print(rf_cm)
rf_df <- as.data.frame(rf_cm$table)

# Plot the confusion matrix
ggplot(data = rf_df, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), vjust = 1) +
  theme_minimal() +
  labs(title = "Confusion Matrix - Random Forest",
       x = "Actual",
       y = "Predicted")

# Perform PCA
pca <- prcomp(scaled_features, scale = TRUE)

# Visualize PCA
fviz_pca_biplot(pca, geom = "point", habillage = Features$emotion_id, addEllipses = TRUE) +
  labs(title = "PCA of Features by Emotion")

# Train an SVM classifier with 10-fold cross-validation
svm_model <- train(emotion_id ~ ., data = trainData, method = "svmLinear", trControl = rf_control)

# Predict on test data
svm_predictions <- predict(svm_model, newdata = testData)

# Confusion matrix
svm_cm <- confusionMatrix(svm_predictions, testData$emotion_id)
print(svm_cm)

# Train a k-NN classifier with 10-fold cross-validation
knn_model <- train(emotion_id ~ ., data = trainData, method = "knn", trControl = rf_control, tuneLength = 5)

# Predict on test data
knn_predictions <- predict(knn_model, newdata = testData)

# Confusion matrix
knn_cm <- confusionMatrix(knn_predictions, testData$emotion_id)
print(knn_cm)


# Train a GBM classifier with 10-fold cross-validation
gbm_model <- train(emotion_id ~ ., data = trainData, method = "gbm", trControl = rf_control, verbose = FALSE)

# Predict on test data
gbm_predictions <- predict(gbm_model, newdata = testData)
gbm_class <- apply(gbm_predictions, 1, which.max)

# Confusion matrix
gbm_cm <- confusionMatrix(as.factor(gbm_class), testData$emotion_id)
print(gbm_cm)


# Train a decision tree classifier with 10-fold cross-validation


tree_model <- train(emotion_id ~ ., data = trainData, method = "rpart", trControl = rf_control)

# Predict on test data
tree_predictions <- predict(tree_model, newdata = testData)



# Confusion matrix
tree_cm <- confusionMatrix(tree_predictions, testData$emotion_id)
print(tree_cm)

# Train a Naive Bayes classifier with 10-fold cross-validation
nb_model <- train(emotion_id ~ ., data = trainData, method = "naive_bayes", trControl = rf_control)

# Predict on test data
nb_predictions <- predict(nb_model, newdata = testData)

# Confusion matrix
nb_cm <- confusionMatrix(nb_predictions, testData$emotion_id)
print(nb_cm)



###################################################################


# Load necessary libraries
library(caret)
library(rpart)

# Set up cross-validation control
rf_control <- trainControl(method = "cv", number = 10)

# Define a grid of hyperparameters
# The complexity parameter (cp) is the primary hyperparameter for rpart
tune_grid <- expand.grid(cp = seq(0.001, 0.1, by = 0.005))

# Train the model with the tuning grid
tree_model <- train(
  emotion_id ~ .,        # The formula for the model
  data = trainData,      # Training data
  method = "rpart",      # Method to use, rpart for decision tree
  trControl = rf_control,# Cross-validation control
  tuneGrid = tune_grid   # Grid of parameters to tune
)

# Print the best tuning parameter
print(tree_model$bestTune)

# Predict on test data
tree_predictions <- predict(tree_model, newdata = testData)

# Evaluate the model performance
confusionMatrix(tree_predictions, testData$emotion_id)




# Load necessary libraries
library(caret)
library(rpart)
set.seed(120)

# Set up cross-validation control
rf_control <- trainControl(method = "cv", number = 10)

# Define a grid of hyperparameters
tune_grid <- expand.grid(cp = seq(0.001, 0.1, by = 0.005))

# Train the model with the tuning grid and custom control parameters
tree_model <- train(
  emotion_id ~ .,
  data = trainData,
  method = "rpart",
  trControl = rf_control,
  tuneGrid = tune_grid,
  control = rpart.control(maxdepth = 10, minsplit = 5, minbucket = 2)
)

# Print the best tuning parameter
print(tree_model$bestTune)

# Predict on test data
tree_predictions <- predict(tree_model, newdata = testData)

# Evaluate the model
confusionMatrix(tree_predictions, testData$emotion_id)



