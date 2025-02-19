theme: jekyll-theme-minimal
## Introduction
This project aims to predict residential home prices using machine learning techniques. It uses a dataset containing various features of homes, such as size, location, and other attributes, to predict the sale price of a house.

## Project Overview

Making home price predictions accessible to everyone can benefit millions of buyers and sellers involved in residential real estate transactions. These predictions help ensure that deals reflect a property’s fair value. As a result, companies like Redfin and Zillow offer an estimation of residential real estate prices. The sale price of a house often depends on various features, such as its size; for instance, a house with a larger lot size may be priced higher than others in the same neighborhood.
        
Machine learning provides various approaches to help build a prediction model that estimates house prices using such features. Building the model requires a training dataset, which helps the model to learn the mapping between the combination of features' values to a home price.\footnote{In addition to the training dataset, machine learning requires a testing dataset to evaluate the accuracy of the model on unseen (i.e., testing) data.} Formally, given $n$ features (i.e., $x_{1}, x_{2}, ... x_{n}$) then the model can be defined as a function \( F \) such that:

$$ 
F(x_1, x_2, ..., x_n) \mapsto Y \quad \text{where} \quad Y \in \mathbb{R}
$$

However, it is hard to find such a dataset. Conveniently, Kaggle (an online platform) provides a public house price dataset with features describing different aspects of residential homes in Ames, Iowa. This project uses the same dataset to build various models using different machine learning approaches such as linear regression, regularization, and ensemble methods. Additionally, the project compares the performance of the models trained using these machine learning approaches, analyzes their strengths and weaknesses, and provides the implementation details for these approaches. [Kaggle Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Algorithms Used

1. Linear Regression

2. Regularization

3. Gradient Boosting & eXtreme Gradient Boosting (XGBoost)

4. Hyperparameter Tuning

5. Ensemble Learning

## Requirements

- Python 3.7+
- Libraries:
  - `pandas`
  - `numpy`
  - `xgboost`
  - `scikit-learn`
  - `matplotlib`
  - `plotly`
  - `statsmodels`

## Project Report
[Residential Price Estimation Using Machine Learning](https://drive.google.com/file/d/1RA-xQpM6f2mc3AqBhQWx-JP2bOgA_gv7/view?usp=sharing)
