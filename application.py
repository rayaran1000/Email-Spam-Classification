from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app=application

#Route for home page

@app.route('/')
def index():
    return render_template('index.html') # Defining the Index Html Page

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html',results=' ') # Home.html will contain fields for getting out Input fields
    else:
        data=CustomData( # Here we are getting all the Input values from the webpage
            Message = request.form.get('Message')           
        )

        pred_df=data.get_data_as_data_frame() # We are getting the dataframe here
        print(pred_df.columns) 

        # Calling the PredictPipeline
        predict_pipeline=PredictPipeline()
        message_category = predict_pipeline.predict(pred_df) 
        return render_template('home.html',results=message_category)
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True) # Maps with 127.0.0.1