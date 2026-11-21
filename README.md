# Predicting Heart Disease From Clinical Attributes

Author: Methuki Pehansa De Silva

## Overview

This project builds an end-to-end machine learning pipeline that predicts whether a patient has heart disease from 14 clinical attributes. It uses a single algorithm throughout, a Random Forest classifier, so that every change in performance comes from data preparation and optimisation and not from switching models. It shows how each stage of improvement changes the results:

Baseline model (rows with missing values deleted)
Improved model (imputation, encoding and scaling)
Tuned model (5-fold cross-validation and GridSearchCV)
Feature-reduced model (PCA)
Problem definition
Item	Detail
Problem	Predict whether a patient has heart disease from clinical measurements
ML task	Binary classification
Target	target, derived from the original num column (num > 0 becomes 1, otherwise 0)
Algorithm	Random Forest (sklearn.ensemble.RandomForestClassifier)
Main metrics	Accuracy, precision, recall (sensitivity), specificity, F1-score, ROC-AUC

Recall (sensitivity) is treated as especially important, because missing a patient who has the disease costs more than a false alarm.

## Dataset
Source: UCI Heart Disease dataset, file heart_disease_uci.csv https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
Size: 920 patient records and 16 columns
Sites: four hospitals (Cleveland, Hungary, Switzerland, VA Long Beach), stored in the dataset column
Class balance: 509 with disease (55%) and 411 without (45%)
Sex balance: 726 male (79%) and 194 female (21%)

## Pipeline
### 1. Exploration, baseline and improved models

Exploration. Missing values, target distribution, disease status by sex, and age distribution by disease status.

### Baseline model.

Drop every row with a missing value (920 rows down to 299, 32.5% retained).
One-hot encode the categorical features (fbs and exang mapped directly to 0/1).
80/20 train/test split, stratified on target and sex combined (239 train, 60 test).
Train a default Random Forest.

Dropping rows also shifted the sex balance from 79/21 to 68/32, so deletion biases the sample towards the hospitals with more complete records.

### Improved model.

Impute missing values: median for numeric columns (age, trestbps, chol, thalch, oldpeak, ca) and mode for fbs, restecg, exang, slope and thal.
Keep all 920 patients and encode to 21 features.
80/20 stratified split (736 train, 184 test).
Scale with StandardScaler, fitted on the training set only.
Train a default Random Forest, and plot feature importances.

The top features were chol, thalch (maximum heart rate), oldpeak (ST depression) and age.

### 2. Cross-validation and hyperparameter tuning
Cross-validation: 5-fold StratifiedKFold on the training set only. The test set is left untouched until final evaluation.

### 3. Feature reduction (PCA)
PCA was fitted on the scaled training set to keep at least 90% of the variance.
14 components were needed, down from 21 features.
The Random Forest was retrained on the components using the best parameters from the tuning stage, so the comparison isolates the effect of feature reduction.
### 4. Model comparison and reflection

All four models are compared on the same metrics, with a bar chart of test accuracy and confusion matrices for each stage.

. Model comparison and reflection

All four models are compared on the same metrics, with a bar chart of test accuracy and confusion matrices for each stage.

Results

Test-set performance

Model	Accuracy	Precision	Recall	Specificity	F1	ROC-AUC
Baseline (raw, rows dropped)	0.8167	0.7931	0.8214	0.8125	0.8070	0.8795
Improved (imputed and scaled)	0.8533	0.8378	0.9118	0.7805	0.8732	0.9214
Tuned (GridSearchCV)	0.8696	0.8421	0.9412	0.7805	0.8889	0.9320
PCA-reduced	0.8859	0.8584	0.9510	0.8049	0.9023	0.9442

5-fold cross-validation (training set)

Model	Mean CV accuracy	Std. dev.
Improved (default parameters)	0.8017	0.0297
Tuned	0.8207	0.0340
PCA-reduced	0.8125	0.0229

## Key findings
Best test score: the PCA-reduced model has the highest test accuracy (0.886), F1 (0.902) and ROC-AUC (0.944).
Best CV score: the tuned model has a slightly higher cross-validation accuracy (0.821 against 0.813), so the tuned and PCA models are closely matched.
Data preparation mattered most. Imputation kept all 920 patients and improved accuracy, precision and recall over the baseline.
Tuning helped. Limiting tree depth and requiring more samples per split raised mean CV accuracy from 0.802 to 0.821.
Overfitting: the test scores (0.87 to 0.89) sit above the CV scores (0.80 to 0.82). This is not classic overfitting, and the 184-patient test set is probably slightly optimistic, so the CV figures are the safer estimate.
