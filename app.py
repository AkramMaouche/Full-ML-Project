from flask import Flask,render_template,request
import pandas as pd  
import numpy as np 
from piplines.predict_pipline import Customdata,PredictPiplines



from sklearn.preprocessing import StandardScaler


application = Flask(__name__,static_url_path='/static')

app = application

#route 
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/predictData',methods = ['GET','POST'])
def predict_datapoint(): 
    if request.method == 'GET':
        return render_template('home.html')
    else: 
        data = Customdata(
            gender=request.form.get('gender'),# get them from the home html page 
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df = data.get_data_as_df()
        print(pred_df)

        predict_pipilene = PredictPiplines()
        results = predict_pipilene.predict(pred_df)
        results = results.round(2)
        return render_template('home.html',results = results[0])


if __name__=="__main__": 
    app.run(host="0.0.0.0",debug=False)