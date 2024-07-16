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
library(ggpubr)  # For QQ plot
library(FSA)  # For Dunn's test
library(vegan)  # For PERMANOVA


required_packages <- c("FSA", "vegan", "ggpubr", "multcomp", "cluster", "caret", "factoextra", "car", "ggplot2", "dplyr", "readr", "tidyr")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}



# Read the data from CSV files
#Features <- read_csv("Feature_28.csv")
Features <- read_csv("All_features.csv")

# Convert necessary columns to factors
Features <- Features %>%
  mutate(emotion_id = as.factor(emotion_id),
         arousal_class = as.factor(arousal_class),
         valence_class = as.factor(valence_class),
         pn = as.factor(pn),
         quadrant = as.factor(quadrant),
         round = as.factor(round))


required_columns <- c("mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
                      "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10",
                      "mfcc_11", "mfcc_12", "mfcc_13", "spectral_centroid",
                      "spectral_bandwidth", "zero_crossing_rate", "rmse",
                      "mean_pressure", "max_pressure",
                      "pressure_variance", "pressure_gradient", "median_force",
                      "iqr_force" , "contact_area",
                      "rate_of_pressure_change","pressure_std",'num_touches','max_touch_duration',
                      "min_touch_duration", 'mean_duration')

# required_columns <- c("mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
#          "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10",
#          "mfcc_11", "mfcc_12", "mfcc_13", "spectral_centroid",
#          "spectral_bandwidth", "rmse",
#          "pitch", "mean_pressure", "max_pressure",
#          "pressure_variance",  "pressure_gradient", "median_force",
#          "iqr_force" ,"touch_duration", "contact_area", "min_force",
#          "rate_of_pressure_change","pressure_std")


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
scaled_combined_features <- cbind(scaled_combined_features,
                                  quadrant = Features$quadrant,
                                  emotion_id = Features$emotion_id,
                                  arousal_class = Features$arousal_class,
                                  valence_class = Features$valence_class,
                                  pn = Features$pn)


# Perform PERMANOVA using the adonis function from the vegan package
permanova_results <- adonis(scaled_combined_features[, required_columns] ~ emotion_id+ pn+emotion_id*pn, data = scaled_combined_features, method = "euclidean")

#permanova_results <- adonis(scaled_combined_features[, required_columns] ~ emotion_id, data = scaled_combined_features, method = "euclidean")


# permanova_results
summary(permanova_results)
print(permanova_results)


# Extracting values from the aov.tab
SS_effect <- permanova_results$aov.tab[1, "SumsOfSqs"]
SS_residual <- permanova_results$aov.tab[2, "SumsOfSqs"]
DF_effect <- permanova_results$aov.tab[1, "Df"]
DF_residual <- permanova_results$aov.tab[2, "Df"]

# Calculating mean squares
MS_effect <- SS_effect / DF_effect
MS_residual <- SS_residual / DF_residual

# Calculating the F-value
F_value <- MS_effect / MS_residual

F_value

# Extract the ANOVA table
aov_tab <- permanova_results$aov.tab

# View the ANOVA table
print(aov_tab)

# Perform PERMANOVA using the adonis function from the vegan package
# permanova_results <- adonis(scaled_combined_features[, required_columns] ~ emotion_id, data = scaled_combined_features, method = "euclidean")
# # permanova_results
# summary(permanova_results)
# print(permanova_results)


set.seed(100)

# Define the pairwise.adonis function
pairwise.adonis <- function(x,factors, sim.method = "euclidean", p.adjust.m = "bonferroni", ...) {
  library(vegan)
  co = combn(unique(as.character(factors)), 2)
  pairs = c()
  F.Model = c()
  R2 = c()
  p.value = c()

  for(elem in 1:ncol(co)) {
    ad = adonis(x[factors %in% c(co[1,elem],co[2,elem]),] ~ factors[factors %in% c(co[1,elem],co[2,elem])], method = sim.method, ...)
    pairs = c(pairs,paste(co[1,elem],'vs',co[2,elem]))
    F.Model = c(F.Model,ad$aov.tab[1,4])
    R2 = c(R2,ad$aov.tab[1,5])
    p.value = c(p.value,ad$aov.tab[1,6])
  }
  p.adjusted = p.adjust(p.value, method = p.adjust.m)
  pairw.res = data.frame(pairs, F.Model, R2, p.value, p.adjusted)
  return(pairw.res)
}

# Run pairwise comparisons
pairwise_results <- pairwise.adonis(scaled_combined_features[, required_columns], scaled_combined_features$emotion_id)
pairwise_results


