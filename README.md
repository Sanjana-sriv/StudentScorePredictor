# 📊 Student Score Predictor

This project implements a simple linear regression model to **predict a student's final grade** based on key academic factors. The model uses **hours studied**, **previous scores**, and **attendance** as input features to estimate the **final percentage/grade**.

## 📁 Dataset

The dataset (`students.csv`) includes the following features:

- `Hours_Studied`: Number of hours the student studied
- `Previous_Scores`: Average of previous academic scores
- `Attendance`: Class attendance rate
- `Final_Grade`: Actual final percentage scored by the student (target variable)

## ⚙️ Model Details

- **Normalization**: Features are scaled using **Min-Max Normalization**.
- **Model Type**: Linear Regression (implemented from scratch using NumPy)
- **Training**: Uses **Gradient Descent** for optimization
- **Evaluation**: The model performance is measured using **Mean Squared Error** and **R² Score**

## 🧠 How It Works

1. Preprocesses the dataset and normalizes features
2. Adds a bias term to the input matrix
3. Trains a linear regression model using gradient descent
4. Plots the convergence of the cost function over iterations
5. Predicts final grades for new student inputs
6. Evaluates the model with the R² score

## 📈 Example Prediction

```python
new_student = np.array([1, 0.8, 0.7, 0.9])  # Bias + Normalized Hours, Scores, Attendance
predicted_grade = predict(new_student, theta)
```

## ✅ Results

- The model trains successfully with decreasing cost per iteration
- R² Score is printed at the end to evaluate accuracy

## 📦 Requirements

- `numpy`
- `pandas`
- `matplotlib`

Install via:

```bash
pip install numpy pandas matplotlib
```
