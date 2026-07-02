# Smart Lender: Loan Eligibility Prediction

An end-to-end Machine Learning web application built using Python, Flask, and XGBoost to predict whether an applicant is eligible for a loan based on various parameters like income, credit history, education, etc.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Model:** XGBoost Classifier
* **Libraries:** Pandas, NumPy, Scikit-learn, Pickle
* **Web Framework:** Flask
* **Frontend:** HTML, CSS

## 🚀 Project Architecture
1. **Data Preprocessing & Training (`train.py`):** Handles missing values, encodes categorical variables, and trains the XGBoost model. Saves the trained model as `xgb_loan_model.pkl`.
2. **Backend Application (`app.py`):** A Flask server that loads the trained model and processes user inputs from the web UI to return real-time predictions.
3. **Frontend UI (`templates/`):** Interactive HTML forms to capture applicant details.
