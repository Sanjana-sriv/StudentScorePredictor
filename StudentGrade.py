import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("students.csv")

# Display first few rows
print(df.head())

# Feature Scaling (Min-Max Normalization)
def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

df[['Hours_Studied', 'Previous_Scores', 'Attendance']] = df[['Hours_Studied', 'Previous_Scores', 'Attendance']].apply(normalize)

# Extract features (X) and target (y)
X = df[['Hours_Studied', 'Previous_Scores', 'Attendance']].values
y = df['Final_Grade'].values

# Add Bias Term (X0 = 1)
X = np.c_[np.ones(X.shape[0]), X]  # Adding ones column for bias

# Hypothesis function (Linear Model)
def predict(X, theta):
    return np.dot(X, theta)

# Cost Function (Mean Squared Error)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# Gradient Descent Algorithm
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        gradient = (1 / m) * np.dot(X.T, (predict(X, theta) - y))
        theta -= alpha * gradient  # Update theta
        cost_history.append(compute_cost(X, y, theta))

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost_history[-1]}")

    return theta, cost_history

# Initialize Parameters
theta = np.zeros(X.shape[1])  # Initialize weights with zeros
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations

# Train the model
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

plt.plot(range(len(cost_history)), cost_history, color='blue')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()

# Example: Predict Final Grade for a new student
new_student = np.array([1, 0.8, 0.7, 0.9])  # Bias term + Normalized Features
predicted_grade = predict(new_student, theta)
print("Predicted Final Grade:", predicted_grade)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

y_pred = predict(X, theta)
print("Model R² Score:", r2_score(y, y_pred))
