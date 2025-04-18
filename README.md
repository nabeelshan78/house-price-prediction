# 🏠 California Housing Price Prediction

This repository presents a complete end-to-end machine learning solution for predicting California housing prices based on demographic, geographic, and economic features. The project leverages advanced feature engineering, custom data transformations, automated pipelines, and ensemble modeling to deliver strong performance and a reproducible workflow.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Workflow Summary](#workflow-summary)  
4. [Notebook Structure](#notebook-structure)  
5. [Models Used](#models-used)  
6. [Evaluation Metrics](#evaluation-metrics)  
7. [Results](#results)  
8. [Project Structure](#project-structure)  

---

## 🚀 Project Overview

- Predict median house values using the **California Housing Dataset**.
- Applied complete ML pipeline including:
  - Data overview and stratified sampling
  - Exploratory data analysis and visualizations
  - Advanced feature engineering (binning, cluster similarity, RBF kernels, encoding, creating new ratio features)
  - Custom preprocessing pipeline
  - Model training, evaluation, and fine-tuning
  - Ensemble modeling using `BaggingRegressor`, `VotingRegressor` and `StackingRegressor`
  - Final model saving and prediction on test data

---

## 🗂️ Dataset

- Source: [California Housing dataset]
- Features include: median income, housing median age, total rooms, bedrooms, population, households, latitude, longitude, etc.
- Target: `median_house_value`

---

## 🔁 Workflow Summary

✔ Data loading and initial overview  
✔ Stratified sampling based on income categories  
✔ Detailed EDA for numeric and categorical features  
✔ Feature engineering using:
- Ratio and interaction terms
- Cluster similarity scores
- RBF kernel transformation for peak detection
- Custom binning strategies

✔ Built complete preprocessing pipeline using `Pipeline`, `make_pipeline` and `ColumnTransformer`  
✔ Model training and comparison of various regressors  
✔ Fine-tuning top models with `RandomizedSearchCV`  
✔ Final model: Voting ensemble of Gradient Boosting and Random Forest  
✔ Saved model and pipeline for future inference

---

## 📒 Notebook Structure

### 1. Data Loading and Overview  
- Summary stats, null checks, data types

### 2. Stratified Sampling of Data  
- To preserve distribution of most important feature `median_income` in train/test split

### 3. EDA and Feature Engineering  
- Univariate, bivariate, and correlation analysis

#### 3.1 Numeric Features  
- Distribution plots, transformations

#### 3.2 Categorical Features  
- One-hot encoding, target analysis

### 4. Data Preprocessing  
- Built custom transformers and feature interaction logic

#### 4.1 Ratio Transformer  
- Features like rooms per household, etc.

#### 4.2 Cluster Similarity  
- KMeans similarity scores

#### 4.3 RBF Similarity Transformation  
- Highlight regions with strong histogram peaks

#### 4.4 Custom Binning Transformer  
- Domain-based binning of skewed features

### 5. Complete Preprocessing Pipeline  
- All preprocessing steps wrapped into a single pipeline  
- Easily reusable for training and inference

### 6. Model Training  
- Trained and compared multiple regressors

#### 6.1 Fine-Tuning Best Models  
- Gradient Boosting and Random Forest via `RandomizedSearchCV`

#### 6.1.3 Voting Regressor  
- Combined top 2 models

#### 6.1.4 Stacking Regressor  
- Layered model for experimentation

### 7. Final Model  
- VotingRegressor selected as best performer

### 8. Prediction  
- Loaded saved model + pipeline, and made predictions on test/new data

---

## 🧠 Models Used

- `Ridge`, `Lasso`, `KNeighborsRegressor`, `DecisionTreeRegressor`  
- `RandomForestRegressor`, `GradientBoostingRegressor`, `AdaBoostRegressor`  
- `ExtraTreesRegressor`, `BaggingRegressor`, `HistGradientBoostingRegressor`  
- `XGBRegressor`, `VotingRegressor`, `StackingRegressor`

---

## 📊 Evaluation Metrics

- **R² Score** (Coefficient of Determination)  
- **MAE** (Mean Absolute Error)  
- **MSE** (Mean Squared Error)  
- **RMSE** (Root Mean Squared Error)  
- Used **Cross-Validation** for robust performance estimates
  
Here are the performance results of the models tested:

![Model Comparison](https://github.com/user-attachments/assets/d770e0f2-4b52-4e8b-8345-977afe45be60)

![image](https://github.com/user-attachments/assets/513ed44b-c587-455f-8e03-4affac58623c)

---

## 🏁 Final Results

After training and fine-tuning various models, and further tuning best ones, the final performance of the **Voting Regressor** (combining `RandomForestRegressor` and `GradientBoostingRegressor`) is as follows:

| Metric       | Train Value   | Test Value   |
|---------------|---------------|--------------|
| **RMSE**      | 34,555.35     | 45,135.07    |
| **R² Score**  | 0.91          | 0.85         |

Final model: `VotingRegressor(RandomForest, GradientBoosting)`
- **Train RMSE**: The root mean squared error on the training data is **34,555.35**, indicating the model's prediction error during training.
- **Test RMSE**: The root mean squared error on the test data is **45,135.07**, reflecting the model's prediction error on unseen data.
- **Train R²**: The R² score on the training data is **0.91**, indicating that the model explains 91% of the variance in the training data.
- **Test R²**: The R² score on the test data is **0.85**, showing that the model explains 85% of the variance in the test data.

These results indicate good generalization to unseen data, with the model performing well both during training and on the test set.

---

## 📁 Project Structure
```bash
├── dataset/                   # dataset (Raw and Prepared)
├── model                      # Saved pipeline and model
├── price_prediction           # EDA & model training notebook
└── README.md                  # Project documentation
