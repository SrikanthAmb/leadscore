import json
import pickle
from flask import Flask, render_template,request,jsonify, url_for
import numpy as np
import pandas as pd

app=Flask(__name__)

# Load the model
log_model=pickle.load(open("logmodel.pkl",'rb'))

@app.route('/',methods=['GET'])

def home():
    return render_template('home.html')

@app.route("/predict",methods=['POST'])

def predict():
    input_features=[float(x) for x in request.form.values()]
    data=[np.array(input_features).reshape(-1,1)]
    output=log_model.predict(data)
    def zen(x):
        if x==1:
            s='Yes'
        else:
            s='No'
        return s

    Answer=zen(int(output[0]))
    
    return render_template('home.html',prediction_text="Will student become Xian ? {}".format(Answer))

if __name__=="__main__":
    app.run(debug=True)