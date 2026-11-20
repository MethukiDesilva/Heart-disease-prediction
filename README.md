Overview

This project builds an end-to-end machine learning pipeline that predicts whether a patient has heart disease from 14 clinical attributes. It uses a single algorithm, a Random Forest classifier, and shows how each stage of improvement changes its performance:

Baseline model (rows with missing values deleted)
Improved model (imputation, encoding and scaling)
Tuned model (5-fold cross-validation and GridSearchCV)
Feature-reduced model (PCA)

The pipeline follows the Portfolio 4 brief: define a problem, implement one algorithm, improve it with cross-validation, hyperparameter tuning and feature reduction, then compare and reflect on the results.

Problem definition
Item	Detail
Problem	Predict whether a patient has heart disease from clinical measurements
ML task	Binary classification
Target	target, derived from the original num column (num > 0 becomes 1, otherwise 0)
Algorithm	Random Forest (sklearn.ensemble.RandomForestClassifier)
Main metrics	Accuracy, precision, recall (sensitivity), specificity, F1-score, ROC-AUC

Recall (sensitivity) is treated as especially important, because missing a patient who has the disease costs more than a false alarm.
