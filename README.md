# Convey-emotion

## Introduction

**Convey-emotion is a research project for analysing affective touch, gestures, and intentions during human–robot interaction with the Pepper robot. The repository includes scripts for feature extraction, statistical analysis, reliability analysis, and machine-learning-based classification.**

**The project aims to support the analysis of how human emotions and social touch gestures can be expressed, measured, and classified through tactile interaction data.**

## Repository Structure

- **All_features.csv**: This file contains the dataset with all the features extracted for analysis.
- **Gesture_and_intention_explanation.docx**: A document explaining the gestures and intentions considered in this study.
- **ICC.R**: An R script for calculating the Intraclass Correlation Coefficient (ICC) to assess the reliability of measurements.
- **PERMANOVA_analysis.R**: An R script for performing PERMANOVA (Permutational Multivariate Analysis of Variance) to analyze the differences between emotions.
- **README.md**: This file.
- **features.py**: A Python script for feature extraction from the dataset.
- **machine_learning.R**: An R script for implementing machine learning models to predict and classify emotions based on the dataset.
- **Emotion_classification**: **A folder containing files and scripts related to emotion classification.**
- **Gesture_classification**: **A folder containing files and scripts related to gesture classification.**

## Data Availability

**The dataset is not publicly uploaded to this repository due to privacy and ethical requirements related to participant data. The data may be shared upon reasonable request by contacting the authors, subject to applicable institutional and ethical requirements.**

## Getting Started

### Prerequisites
- R 
- Python 
- Required Python libraries: pandas, numpy, scikit-learn
- Required R libraries: vegan, lme4, caret

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/qiaoqiao2323/Convey-emotion.git
   cd Convey-emotion
