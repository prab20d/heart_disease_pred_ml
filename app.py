from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('heart_disease_pred_rf_p.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        age = int(request.form['age'])
        sex=float(request.form['sex'])
        trestbps=int(request.form['trestbps'])

        
        chol = int(request.form['chol'])
        thalach=float(request.form['thalach'])
        oldpeak=int(request.form['oldpeak'])

        
        target = int(request.form['target'])
        cp_1=float(request.form['cp_1'])
        cp_2=float(request.form['cp_2'])
        cp_3=float(request.form['cp_3'])
        fbs_1=int(request.form['fbs_1'])

        restecg_1 = int(request.form['restecg_1'])
        restecg_2=float(request.form['restecg_2'])
        exang_1=float(request.form['exang_1'])
        slope_1=float(request.form['slope_1'])
        slope_2=int(request.form['slope_2'])

        ca_1 = int(request.form['ca_1'])
        ca_2=float(request.form['ca_2'])
        ca_3=float(request.form['ca_3'])
        ca_4=float(request.form['ca_4'])

        thal_1 = int(request.form['thal_1'])
        thal_2=float(request.form['thal_2'])
        thal_3=float(request.form['thal_3'])

        prediction=model.predict([[age,sex, trestbps, chol,thalach,oldpeak, 
        target,
       cp_1, cp_2, cp_3, fbs_1, 
       restecg_1, restecg_2, 
       exang_1,
       slope_1, slope_2, 
       ca_1, ca_2, ca_3, ca_4,
        thal_1,
       thal_2, thal_3]])
        output=round(prediction[0],2)
        if output==0:
            return render_template('index.html',prediction_texts="you are not having heart disease")
        else:
            return render_template('index.html',prediction_text="You are having heart disease {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

