import sys
import pandas as pd
import numpy as np
import os
import pickle
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, redirect, request


# Create an instance of Flask
app = Flask(__name__)

with open(f'best_model.pkl', "rb") as f:
    saved_model = pickle.load(f)

# Route to render index.html template using data from Mongo
@app.route("/", methods=["GET", "POST"])
def home():

    output_message = ""

    if request.method == "POST":
        person_age = float(request.form["person_age"])
        person_income = float(request.form["person_income"])
        person_home_ownership = float(request.form["person_home_ownership"])
        person_emp_length = float(request.form["person_emp_length"])
        loan_intent = float(request.form["loan_intent"])
        loan_grade = float(request.form["loan_grade"])
        loan_amnt = float(request.form["loan_amnt"])
        loan_int_rate = float(request.form["loan_int_rate"])
        loan_percent_income = float(request.form["loan_percent_income"])
        cb_person_default_on_file = float(request.form["cb_person_default_on_file"])
        cb_person_cred_hist_length = float(request.form["cb_person_cred_hist_length"])
        print(person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length)


    # data must be converted to df with matching feature names before predict

        columns = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_intent', 'loan_grade',
            'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']
        
        data = pd.DataFrame(np.array([[person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade,
            loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length]]), columns=columns)


        #Scale new data from inputted values from the form
        #X_scaler = StandardScaler().fit(data)
        #X_new_scaled = X_scaler.transform(data)

        result = saved_model.predict(data)

        if result == 1:
            output_message = "This loan is at high-risk for a default."
        else:
            output_message = "This loan is NOT at high-risk for a default."

        return render_template("index.html", message = output_message)

    return render_template("index.html", message = "")




def isnumber(x):
    if x is None: return(False)
    if pd.isnull(x): return(False)
    try:
        int(x)
        return(True)
    except ValueError:
        try:
            float(x)
            return(True)
        except ValueError:
            return(False)
    return(False)
    





if __name__ == "__main__":
    app.run(debug=True)
    