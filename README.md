# Car Auto Pricer

## Overview
This project focuses on predicting car prices based on common features using different prediction models. The goal is to analyze, clean, and train models on a dataset to accurately predict car prices. The results from different models are compared to determine the best approach.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Analysis](#data-analysis)
3. [Data Processing](#data-processing)
4. [Modeling](#modeling)
5. [Models Explanation](#models-explanation)
6. [Our Interface](#our-interface)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction
This project aims to navigate the automotive marketplace by building a predictive model to estimate car prices based on various attributes. The process involves:
- Data collection and cleaning
- Data analysis to uncover patterns
- Training multiple prediction models
- Comparing model performances

## Data Analysis
The initial analysis revealed missing values and inconsistencies in the dataset, such as the "Doors" column, which was dropped. Key steps included identifying highly correlated features with price and visualizing data through bar charts and histograms.

## Data Processing
Steps for data cleaning and preprocessing included:
1. Loading the dataset (`train.csv`).
2. Summarizing key information about the dataset.
3. Handling missing values in columns like 'Levy' and 'Mileage'.
4. Filtering rows based on 'Price'.
5. Removing irrelevant columns like 'ID' and 'Doors'.
6. Converting text columns to lowercase.
7. Label encoding categorical columns.

## Modeling
Several regression models were used to predict car prices:
- **Decision Tree Regression:** Captures complex patterns but may overfit.
- **Random Forest Regression:** Combines multiple trees to improve performance.
- **Gradient Boosting Regression:** Sequentially builds an ensemble of trees to correct errors.
- **Extra Trees Regression:** Introduces randomness in tree construction to enhance diversity.

## Models Explanation
### Gradient Boosting Regressor
A powerful ensemble method that combines predictions from multiple weak learners using gradient descent optimization to minimize error.

### Extra Trees Regressor
An ensemble method that introduces randomness in tree building and uses bootstrapping to reduce overfitting and improve generalization.

## Our Interface
**Autopricer** is a user-friendly web interface that uses advanced machine learning models to predict car prices. Users can input car details, and the interface provides accurate valuations based on multiple models.

### Index Page
![image](https://github.com/mariamAboujenane/Car-Auto-Pricer/assets/106840796/429f2aae-a88c-4b8c-895c-bdccc79087e0)

### Enter Car’s Details
![image](https://github.com/mariamAboujenane/Car-Auto-Pricer/assets/106840796/af92096a-9b4b-42d4-8c30-5084a2914fe0)
![image](https://github.com/mariamAboujenane/Car-Auto-Pricer/assets/106840796/0e43c3b4-b087-45aa-a8d9-bc430c312c3d)

### Price Predictions
![image](https://github.com/mariamAboujenane/Car-Auto-Pricer/assets/106840796/edb86fdf-3746-4803-9097-1b3b1984499d)

### Models' Comparison
![image](https://github.com/mariamAboujenane/Car-Auto-Pricer/assets/106840796/3b448f3f-e372-40fe-836a-3312666f2f87)
![image](https://github.com/mariamAboujenane/Car-Auto-Pricer/assets/106840796/ae1380e5-e402-42d2-99b1-36a0bb77a2ab)
![image](https://github.com/mariamAboujenane/Car-Auto-Pricer/assets/106840796/562ad848-f8e0-456c-8374-a1503854ef67)
![image](https://github.com/mariamAboujenane/Car-Auto-Pricer/assets/106840796/75c419cc-528f-4188-ba9a-875bbad403e2)

## Conclusion
The project successfully developed predictive models to estimate car prices, with the production year and luxury features emerging as strong indicators. The interactive interface and rigorous model comparison provide valuable insights for stakeholders in the automotive industry.

## References
1. [Data Analysis with Python](https://www.geeksforgeeks.org/data-analysis-with-python/)
2. [Python Data Cleaning](https://realpython.com/python-data-cleaning-numpy-pandas/)
3. [Regression Trees in Python](https://python-course.eu/machine-learning/regression-trees-in-python.php)
4. [How to Compare Machine Learning Models](https://neptune.ai/blog/how-to-compare-machine-learning-models-and-algorithms)
5. [Flask Tutorial](https://www.geeksforgeeks.org/flask-tutorial/)

---
