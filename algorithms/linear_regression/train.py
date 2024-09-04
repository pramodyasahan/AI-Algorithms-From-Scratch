import numpy as np
import pandas as pd
from linear_regression import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Fetch the dataset
housing_data = fetch_california_housing()
X = housing_data.data
y = housing_data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_normalized = (X_train - mean) / std
X_test_normalized = (X_test - mean) / std

regressor = LinearRegression()
regressor.fit(X_train_normalized, y_train)
