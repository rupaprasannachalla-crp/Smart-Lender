# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load only the trained XGBoost model file
model = pickle.load(open('xgb_loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from web form
        gender = int(request.form['Gender'])
        married = int(request.form['Married'])
        dependents = int(request.form['Dependents'])
        education = int(request.form['Education'])
        self_employed = int(request.form['Self_Employed'])
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = int(request.form['Property_Area'])
        
        # Create features array for prediction
        features = np.array([[gender, married, dependents, education, self_employed, 
                              applicant_income, coapplicant_income, loan_amount, 
                              loan_amount_term, credit_history, property_area]])
        
        # Make prediction
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            result_text = "Congratulations! You are eligible for the loan."
        else:
            result_text = "Sorry, you are not eligible for the loan."
            
        return render_template('result.html', prediction_text=result_text)

if __name__ == '__main__':
    app.run(debug=True)