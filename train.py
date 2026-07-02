import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# 1. Load Dataset
df = pd.read_csv('loan_data.csv')

# Drop Loan_ID if it exists in data
if 'Loan_ID' in df.columns:
    df = df.drop(columns=['Loan_ID'])

# 2. Separate Numerical and Categorical Columns explicitly
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

# 3. Handling Missing Values properly
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].astype(str).fillna(df[col].mode()[0])

# 4. Handling Categorical Columns using LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 5. Split Features and Target
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Building using XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 7. Save the trained model file
with open('xgb_loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model Training Success! xgb_loan_model.pkl file created.")