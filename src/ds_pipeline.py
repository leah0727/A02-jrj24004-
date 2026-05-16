from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Save boxplot
plt.figure(figsize=(6, 4))
df['MedHouseVal'].plot.box()

plt.title("Boxplot of Median House Value")
plt.ylabel("Median House Value")

plt.tight_layout()

plt.savefig("figures/med_house_value_boxplot.png")

plt.close()

#PR1. 

import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt


# 1. Load California Housing dataset
data = fetch_california_housing(as_frame=True)

X = data.frame.drop(columns=["MedHouseVal"])
y = data.frame["MedHouseVal"]

print(X.head())
print(y.head())


# 2. Split into train / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



##PR2. 

# 4. Train MLPRegressor
# Includes early_stopping=True and custom hyperparameters
mlp = MLPRegressor(
    random_state=42,
    hidden_layer_sizes=(10, 5),
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=300,
    batch_size=1000,
    activation="relu",
    validation_fraction=0.2,
    early_stopping=True
)

mlp.fit(X_train_scaled, y_train)


# 5. Predictions
y_train_pred = mlp.predict(X_train_scaled)
y_test_pred = mlp.predict(X_test_scaled)


# 6. Evaluation
print("Train R2:", r2_score(y_train, y_train_pred))
print("Train MAE:", mean_absolute_error(y_train, y_train_pred))

print("Test R2:", r2_score(y_test, y_test_pred))
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
