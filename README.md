# Diabetes Training Model

## Introduction
This project was developed to apply machine learning techniques in the field of healthcare, specifically to classify patients as diabetic or non-diabetic based on various medical features.

## Objective
The main objectives of this project are to:
- Train a model that can accurately classify patients with diabetes.
- Explore the relationships between different medical features.
- Optimize the performance of the Random Forest model using hyperparameter tuning.

## About the Dataset
The dataset used in this project contains information about various medical attributes of patients, including:
- Pregnancies: Number of times the patient has been pregnant.
- Glucose: Plasma glucose concentration.
- BloodPressure: Diastolic blood pressure (mm Hg).
- SkinThickness: Triceps skin fold thickness (mm).
- Insulin: 2-Hour serum insulin (mu U/ml).
- BMI: Body mass index (weight in kg/(height in m)^2).
- DiabetesPedigreeFunction: Diabetes pedigree function.
- Age: Age of the patient (years).
- Outcome: Class variable (0 or 1) where 1 indicates that the patient has diabetes.

## Steps
1. Importing Libraries: Import necessary libraries such as pandas, numpy, seaborn, matplotlib, and scikit-learn.
2. Loading the Dataset: Load the dataset from a CSV file.
3. Exploring the Data: Perform descriptive statistics and visualize data distribution.
4. Visualizing Correlations: Create a heatmap to visualize correlations between features.
5. Data Splitting: Split the dataset into training and testing sets.
6. Model Training: Train a Random Forest model with hyperparameter tuning using GridSearchCV.
7. Model Evaluation: Evaluate the model using metrics like accuracy, confusion matrix, and classification report.
8. ROC-AUC Curve: Plot the ROC-AUC curve to assess model performance.

## Conclusion
This project serves as a practical implementation of machine learning in healthcare. By following the steps outlined above, a Random Forest model was trained and optimized to classify diabetic patients effectively. All steps in the project are well-documented with comments and explanations.

Feel free to provide any feedback or suggestions.
