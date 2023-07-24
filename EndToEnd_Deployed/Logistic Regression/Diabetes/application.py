from flask import Flask, request, app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application
scaler = pickle.load(open('Model/standardScaler.pkl','rb'))
model = pickle.load(open('Model/modelForPrediction.pkl','rb'))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result = ''
    if request.method == 'POST':
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        
        new_data = scaler.tranform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
        
        if(predict[0]==1):
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'
            
    return render_template('single_prediction.html',result=result)
    

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
    
    
#pip install -r requirements.txt