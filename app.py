from flask import Flask, render_template, redirect
import pandas as pd
import numpy as np
import pickle
from sklearn import tree
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create an instance of Flask
app = Flask(__name__)

with open(f'best_rf_model.pickle', "rb") as f:
    model = pickle.load(f)

feature_names = model.get_booster().feature_names

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

    # if request.method == "POST":
    #     recency = float(request.form["recency"])
    #     frequency = float(request.form["frequency"])
    #     monetary = float(request.form["monetary"])
    #     time = float(request.form["time"])

        # data must be converted to df with matching feature names before predict

        data = pd.DataFrame(np.array([[person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade,
        loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length]]), columns=feature_names)
        result = model.predict(data)
        if result == 1:
            output_message = "This application is at high-risk for a loan default."
        else:
            output_message = "This application is NOT at high-risk for a loan default."
    
    return render_template("index.html", message = output_message)

    #     data = pd.DataFrame(np.array([[recency, frequency, monetary, time]]), columns=feature_names)
    #     result = model.predict(data)
    #     if result == 1:
    #         output_message = "Nice, you will donate soon, thank you ^_^"
    #     else:
    #         output_message = "Please consider donating :-("
    
    # return render_template("index.html", message = output_message)

if __name__ == "__main__":
    app.run()
