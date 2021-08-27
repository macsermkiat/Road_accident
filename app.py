# -*- coding: utf-8 -*-
"""
@author: Sermkiat
"""

import numpy as np
import pandas as pd
import flask
import pickle
import joblib
# app
app = flask.Flask(__name__)

# load model

model = joblib.load('lr_rfe.pkl')
#model = pickle.load(open("lr_rfe.pkl","rb"))



# routes
@app.route("/")
def home():
    return """
           <body> 
           <h1>Welcome to Road accident prediction<h1>
           <a href="/page">Begin</a>
           </body>"""

    
@app.route("/page")
def page():
    with open("page.html", 'r') as viz_file:
        return viz_file.read()


@app.route("/result", methods=["GET", "POST"])
def result():
    """Gets prediction using the HTML form"""
    if flask.request.method == "POST":
        inputs = flask.request.form
        gender = inputs["gender"]
        age = inputs["age"]
        mechan = inputs["mechan"]
        sbp = inputs["sbp"]
        dbp = inputs["dbp"]
        rr = inputs["rr"]
        pr = inputs["pr"]
        osat = inputs["osat"]
        gcs = inputs["gcs"]
    cols = ['gender', 'age', 'mechan', 'sbp', 'dbp', 'pr', 'rr', 'osat', 'gcs']
    X_new = pd.DataFrame(np.array([(gender)] + [(age)] + [(mechan)]+
                     [(sbp)] + [(dbp)] + [(pr)] + 
                     [(rr)] + [(osat)] + [(gcs)]).reshape(1, -1), columns = cols)
    
    y_prob = model.predict_proba(X_new)
    yhat = pd.Series(y_prob[0][1]).map(lambda x: 1 if x> 0.1537 else 0)
    if yhat[0] == 0:
        outcome = "Low probability of dead"
    else:
        outcome = "Significant probability of dead, need intensive management"
    
    #return "NeuralNetwork model predict outcome of " + outcome + "having seroma." 
    return """<body><p> The prabability of death is """ + str(round(y_prob[0][1], 2)) + """</p>
            <p>Predict: """ + outcome + """</p></body>"""
   

if __name__ == '__main__':
    """Connect to Server"""
    #HOST = "127.0.0.1"
    #PORT = "5000"
    app.run(debug=True)
