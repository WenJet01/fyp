# from asyncio.windows_events import NULL

from flask import Flask, render_template, request

from datetime import datetime
import pickle
import pandas as pd
import daal4py
import os
import joblib
import numpy as np



app = Flask(__name__)
model = joblib.load("modelv3.joblib")


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template('index.html')

@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/result", methods=['GET', 'POST'])
def result():

    DF = pd.DataFrame(columns=['NP2_EFFECTDATE', 'PPR_PRODCD', 'NPR_PREMIUM', 'CLF_LIFECD',
       'NSP_SUBPROPOSAL', 'NPR_SUMASSURED', 'NLO_TYPE', 'NLO_AMOUNT',
       'AAG_AGCODE', 'PCL_LOCATCODE', 'OCCUPATION', 'CATEGORY', 'DueCount',
       'NPH_SEX', 'NPH_BIRTHDATE'])

    DF.loc[0,'NPH_BIRTHDATE'] = request.form['birthday'][0:4]
    DF.loc[0,'NPH_SEX'] = request.form['gender']
    DF.loc[0,'CATEGORY'] = request.form['work']
    DF.loc[0,'OCCUPATION'] = request.form['occupation']
    DF.loc[0,'PPR_PRODCD'] = request.form['productCode']
    DF.loc[0,'NLO_TYPE'] = request.form['premiumType']
    DF.loc[0,'NP2_EFFECTDATE'] = request.form['effectiveDate'][0:4]
    DF.loc[0,'NPR_PREMIUM'] = request.form['premiumAmt']
    DF.loc[0,'NPR_SUMASSURED'] = request.form['sumAssure']
    DF.loc[0,'NLO_AMOUNT'] = request.form['extraCharge']
    DF.loc[0,'NSP_SUBPROPOSAL'] = request.form['subproposal']
    DF.loc[0,'CLF_LIFECD'] = request.form['lifecd']
    DF.loc[0,'AAG_AGCODE'] = request.form['agentCode']
    DF.loc[0,'PCL_LOCATCODE'] = request.form['location']
    DF.loc[0,'DueCount'] = request.form['dueCount']

    print(DF)

    prob = model.predict_proba(DF)[:, 1]
    print(prob)
    prob = str(round(prob[0]*100, 2))
    return render_template('result.html', prob = prob)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
    
