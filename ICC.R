# Load necessary libraries
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(car)
library(factoextra)
library(caret)
library(cluster)
library(multcomp)
library(e1071)
library(class)
library(gbm)
library(rpart)
library(irr)

# Read the data from CSV files
Features <- read_csv("All_features.csv")


# Convert necessary columns to factors
Features$emotion_id <- as.factor(Features$emotion_id)
Features$arousal_class <- as.factor(Features$arousal_class)
Features$valence_class <- as.factor(Features$valence_class)
Features$pn <- as.factor(Features$pn)
Features$quadrant <- as.factor(Features$quadrant)
Features$round <- as.factor(Features$round)



required_columns <- c("mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
                      "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10",
                      "mfcc_11", "mfcc_12", "mfcc_13", "spectral_centroid",
                      "spectral_bandwidth", "zero_crossing_rate", "rmse",
                      "mean_pressure", "max_pressure",
                      "pressure_variance", "pressure_gradient", "median_force",
                      "iqr_force" , "contact_area",
                      "rate_of_pressure_change","pressure_std",'num_touches','max_touch_duration',
                      "min_touch_duration", 'mean_duration')

# Check if all required columns are present
missing_columns <- setdiff(required_columns, colnames(Features))
if(length(missing_columns) > 0) {
  stop("The following required columns are missing: ", paste(missing_columns, collapse = ", "))
}

# Combine the specified features
combined_features <- Features %>%
  dplyr::select(all_of(required_columns))

# Scale numeric features
numeric_features <- combined_features %>%
  select_if(is.numeric)
scaled_features <- scale(numeric_features)
scaled_combined_features <- as.data.frame(scaled_features)
scaled_combined_features$quadrant <- Features$quadrant
scaled_combined_features$emotion_id <- Features$emotion_id
scaled_combined_features$arousal_class <- Features$arousal_class
scaled_combined_features$valence_class <- Features$valence_class
scaled_combined_features$pn <- Features$pn


# Combine sound and tactile features
combined_features <- Features %>%
  dplyr::select(all_of(required_columns))

# Standardize the data
scaled_features <- scale(combined_features)

# Perform PCA
pca_result <- prcomp(scaled_features, center = TRUE, scale. = TRUE)

# Summary of PCA to decide the number of components to retain
summary(pca_result)

# Get the principal components
pca_scores <- as.data.frame(pca_result$x)

# Add emotion_id and pn back to the PCA scores
pca_scores$emotion_id <- Features$emotion_id
pca_scores$pn <- Features$pn

# Calculate ICC for the principal components
icc_results <- list()
for (emotion in unique(pca_scores$emotion_id)) {
  emotion_data <- pca_scores %>%
    filter(emotion_id == emotion) %>%
    dplyr::select(starts_with("PC"))  # Select principal components

  # Reshape data for ICC calculation
  emotion_data <- as.data.frame(t(emotion_data))

  # Calculate ICC
  icc_result <- icc(emotion_data, model = "twoway", type = "consistency", unit = "single")

  # Store result
  icc_results[[as.character(emotion)]] <- icc_result
}

# Print ICC results for each emotion
for (emotion in names(icc_results)) {
  cat("\nEmotion:", emotion, "\n")
  print(icc_results[[emotion]])
}

# Visualize ICC results
icc_values <- sapply(icc_results, function(x) x$value)
icc_df <- data.frame(emotion_id = names(icc_values), ICC = icc_values)


ggplot(icc_df, aes(x = emotion_id, y = ICC, fill = emotion_id)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "ICC for Each Emotion", x = "Emotion ID", y = "ICC Value")

















library(dplyr)

icc_results <- list()

# Define the individual features to check
features_to_check <- c("max_pressure", "pressure_variance", "pressure_gradient",
                       "iqr_force", "contact_area", "rate_of_pressure_change",
                       "pressure_std")

# Define MFCC features as a combined feature
mfcc_features <- paste0("mfcc_", 1:13)
#quadrant
# Iterate through each emotion
for (emotion in unique(Features$emotion_id)) {
  emotion_data <- Features %>%
    filter(emotion_id == emotion)

  icc_results[[as.character(emotion)]] <- list()

  # Calculate ICC for MFCCs as a combined feature
  mfcc_data <- emotion_data %>%
    dplyr::select(all_of(mfcc_features))

  icc_result_mfcc <- icc(mfcc_data, model = "twoway", type = "consistency", unit = "single")
  icc_results[[as.character(emotion)]][["mfccs"]] <- icc_result_mfcc

  # Calculate ICC for each individual feature
  for (feature in features_to_check) {
    feature_data <- emotion_data %>%
      dplyr::select(all_of(feature))

    # Calculate ICC if the feature has more than one column (e.g., MFCCs)
    if (ncol(feature_data) > 1) {
      icc_result <- icc(feature_data, model = "twoway", type = "consistency", unit = "single")
    } else {
      icc_result <- icc(cbind(1:nrow(feature_data), feature_data), model = "twoway", type = "consistency", unit = "single")
    }

    icc_results[[as.character(emotion)]][[feature]] <- icc_result
  }
}

# Print ICC results for each emotion and each feature
for (emotion in names(icc_results)) {
  cat("\nEmotion:", emotion, "\n")
  for (feature in names(icc_results[[emotion]])) {
    cat("\nFeature:", feature, "\n")
    print(icc_results[[emotion]][[feature]])
  }
}


icc_values <- sapply(icc_results, function(x) x$value)
icc_df <- data.frame(round = names(icc_values), ICC = icc_values)

ggplot(icc_df, aes(x = round, y = ICC, fill = round)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "ICC for Each Round", x = "Round", y = "ICC Value")



icc_results <- list()

# Visualize ICC results
icc_values <- sapply(icc_results, function(x) x$value)
icc_df <- data.frame(quadrant = names(icc_values), ICC = icc_values)

ggplot(icc_df, aes(x = quadrant, y = ICC, fill = quadrant)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "ICC for Each Emotion", x = "Emotion ID", y = "ICC Value")

# 使用ANOVA评估不同参与者表达不同情绪时的差异
anova_results <- aov(mean_pressure ~ quadrant * pn, data = Features)
summary(anova_results)

# 可视化不同情绪下的特征差异
ggplot(Features, aes(x = quadrant, y = mean_pressure, fill = quadrant)) +
  geom_boxplot() +
  facet_wrap(~ pn) +
  theme_minimal() +
  labs(title = "Mean Pressure Across Different Emotions and Participants", x = "Emotion ID", y = "Mean Pressure")

# 评估表达不同情绪时的差异（使用配对t检验）
pairwise_t_results <- list()

for (feature in required_columns) {
  pairwise_t_result <- pairwise.t.test(Features[[feature]], Features$quadrant, p.adjust.method = "bonferroni")
  pairwise_t_results[[feature]] <- pairwise_t_result
}

# 打印配对t检验结果
pairwise_t_results



####rounds

# 1. Intra-Round Consistency Across Participants (using ICC)
icc_results <- list()

for (rou in unique(Features$round)) {
  round_data <- Features %>%
    filter(round == rou) %>%
    dplyr::select(starts_with("mfcc"), max_pressure, pressure_variance, pressure_gradient)

  icc_result <- icc(round_data, model = "twoway", type = "consistency", unit = "single")
  icc_results[[as.character(round)]] <- icc_result
}

# 打印每轮的ICC结果
icc_results

# Visualize ICC results
icc_values <- sapply(icc_results, function(x) x$value)
icc_df <- data.frame(round = names(icc_values), ICC = icc_values)

ggplot(icc_df, aes(x = round, y = ICC, fill = round)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "ICC for Each Round", x = "Round", y = "ICC Value")

# 2. 使用ANOVA评估不同参与者在不同轮次表达情绪时的差异
anova_results <- aov(mean_pressure ~ round * pn, data = Features)
summary(anova_results)

# 可视化不同轮次的特征差异
ggplot(Features, aes(x = round, y = mean_pressure, fill = round)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Mean Pressure Across Different Rounds and Participants", x = "Round", y = "Mean Pressure")

