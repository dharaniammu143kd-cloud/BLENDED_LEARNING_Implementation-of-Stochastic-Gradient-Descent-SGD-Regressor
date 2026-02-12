# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Sure Dharani 😊✨
Here is the **Algorithm in 4 simple lines** for your record:

**Algorithm:**

1. Import the required libraries and load the dataset.
2. Preprocess the data by removing unnecessary columns and converting categorical variables.
3. Split the dataset into training and testing sets and train the SGD Regressor model.
4. Predict the output and evaluate the model performance using MSE, MAE, and R² scores

## Program:
```
Developed by: DHARANI B 
RegisterNumber:212225230053  
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("CarPrice_Assignment (1).csv")
print(data.head())
print(data.info())

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
print(data.head())
print(data.info())
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))

# Fitting the model on the training data
sgd_model.fit(X_train, y_train)

# Making predictions
y_pred = sgd_model.predict(X_test)

# Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('MAE= ',mean_absolute_error(y_test, y_pred))
print(f"R2: {r2_score(y_test, y_pred):.4f}")

# Print evaluation metrics
print('Name: DHARANI B ')
print('Reg. No: 212225230053 ')
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Print model coefficients
print("\nModel Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

# Visualizing actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()
```

## Output:
<img width="970" height="781" alt="Screenshot 2026-02-12 130915" src="https://github.com/user-attachments/assets/34fa179f-734d-46d3-be85-a5f7bc9d9ba0" />
<img width="892" height="763" alt="Screenshot 2026-02-12 130957" src="https://github.com/user-attachments/assets/d1db273b-59ac-4aa2-a367-4205834b0f0f" />
<img width="821" height="740" alt="Screenshot 2026-02-12 130933" src="https://github.com/user-attachments/assets/45c22e3b-f712-4c5c-bd04-c855c8613d81" />
<img width="809" height="734" alt="Screenshot 2026-02-12 131017" src="https://github.com/user-attachments/assets/2e2fa859-f541-41db-a748-2ac7734b8497" />
<img width="855" height="221" alt="Screenshot 2026-02-12 131101" src="https://github.com/user-attachments/assets/67b91fe9-ac4c-4009-9529-27f75e313e5a" />
<img width="836" height="574" alt="Screenshot 2026-02-12 131118" src="https://github.com/user-attachments/assets/c2366a5f-f384-40ca-ba5c-f6da9d02570e" />




## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
