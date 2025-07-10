# 💼 Salary Prediction using Ensemble Machine Learning Techniques

This project aims to predict employee salaries based on demographic, educational, and professional features using an **Ensemble Learning** approach. It includes model training using scikit-learn and an interactive user interface built with **Streamlit**.

---

## 🚀 Project Overview

- 📊 **Goal:** Predict employee salaries with improved accuracy.
- 🧠 **Approach:** Combine multiple regressors using **Voting Regressor** (Random Forest + Gradient Boosting).
- 🌐 **UI:** Built with **Streamlit** for interactive user input and predictions.
- 💾 **Model Saving:** Exported using `joblib` for reuse and deployment.

---

## 📁 Dataset Features

The model is trained on a dataset containing:

| Feature               | Type        | Description                         |
|----------------------|-------------|-------------------------------------|
| Age                  | Numerical   | Age of the employee                 |
| Gender               | Categorical | Male/Female                         |
| Education Level      | Categorical | Bachelor's, Master's, PhD, etc.     |
| Job Title            | Categorical | e.g., Software Engineer             |
| Years of Experience  | Numerical   | Total work experience in years      |
| Remote Work Status   | Categorical | Remote, On-site, Hybrid             |
| Industry             | Categorical | e.g., IT, Finance, Healthcare       |
| Company Size         | Categorical | Small, Medium, Large                |
| City or Region       | Categorical | Location                            |
| Salary               | Target      | Annual salary (target variable)     |

---

## 🧠 Model Architecture

- **Preprocessing:**
  - One-Hot Encoding for categorical variables
  - Standard Scaling for numeric variables
  - `ColumnTransformer` for column-wise transformations

- **Models:**
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
  - Combined via `VotingRegressor`

- **Pipeline:**
  - Sklearn `Pipeline` used to combine preprocessing and model steps
  - Easily exportable and deployable using `joblib`

---

## 📈 Evaluation Metrics

- **R² Score:** Indicates goodness of fit
- **Mean Squared Error (MSE):** Measures prediction error

---

## 🖥️ Streamlit App Usage

### Install Dependencies 
   - pip install pandas numpy scikit-learn joblib streamlit

### Run Streamlit app
   - streamlit run streamlit_app.py
