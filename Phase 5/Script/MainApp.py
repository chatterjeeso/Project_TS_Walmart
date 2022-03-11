# -*- coding: utf-8 -*-
"""
1: Pick the value from Index.html
2: Pass the value to a function
3. Apply Logistic Regresion, check for 1,0
4. If 0 then close and send reply as 0
5. If 1 then -
    a) check if based on input Date,Store, Item any exsisting data is there or not, if yes pick rest of the column values
    b) If the combination does not exsist then for that Store,Station, Month pick average Weather details, merge and send to predictor
6. Standradize the input
7. Pass it to predictor
8. Pass the prediction back to Web
"""
from flask import Flask, jsonify, request,render_template
import numpy as np
import joblib
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#from sklearn.externals import joblib
#from joblib import dump, load
import os
import warnings
warnings.filterwarnings('ignore')
import flask
app = Flask(__name__, static_url_path='/static')


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/querypage')
def index():
    return flask.render_template('querypage.html')


@app.route('/predict', methods=['POST'])
def predict():
    #setting up input
    passbackInput = request.form.to_dict()#['2012-07-31', '23', '3']#['29-07-2012', '40', '5']
    passbackInput = list(passbackInput.values())
    dateVal  = passbackInput[0]     
    storeVal = int(passbackInput[1])  
    itemVal  = int(passbackInput[2])
    #print(dateVal,storeVal, itemVal)
    
    #------------------------------------------------------------
    # Reading CSV files for Classification model
    path = os.getcwd()
    df_cls = pd.read_csv(os.path.join(path, "df_Classifier_cleand.csv"))
    #------------------------------------------------------------
    #Check if data already exsist in CSV or not
    df_chkRecord = df_cls.loc[(df_cls['date'] == dateVal) & (df_cls['store_nbr'] == storeVal) & (df_cls['item_nbr'] == itemVal)]
    #------------------------------------------------------------
    ActualValue = "New data point- No previous value"
    CheckAvailable = "NOT AVAILABLE"
    #If no data, then pick median data for the same store & same month
    if len(df_chkRecord)==0:
        df_weatherVal1 = df_cls.loc[df_cls['Month']==int(passbackInput[0].split("-")[1])] 
        df_weatherVal2 = df_weatherVal1.loc[df_weatherVal1['store_nbr'] == int(storeVal)]
        df_weatherVal_median = df_weatherVal2.median()
        
        QueryPoint_Dummy = df_weatherVal_median.to_frame()
        columnLists = ["store_nbr","item_nbr","station_nbr","tmax","depart","cool","snowfall","preciptotal","stnpressure","resultspeed","resultdir","avgspeed","Day",
        "Month","Holiday","codesum_BCFG","codesum_BLDU","codesum_BLSN","codesum_BR","codesum_DU","codesum_DZ","codesum_FG","codesum_FG+","codesum_FU",
        "codesum_FZDZ","codesum_FZFG","codesum_FZRA","codesum_GR","codesum_GS","codesum_HZ","codesum_MIFG","codesum_PL","codesum_PRFG","codesum_RA",
        "codesum_SG","codesum_SN","codesum_SQ","codesum_TS","codesum_TSRA","codesum_TSSN","codesum_UP","codesum_VCFG","codesum_VCTS","codesum_nan","units"]
        values = [] 
        for clmn in columnLists:
            values.append(QueryPoint_Dummy[0][clmn])
    
        #Final data point which will be passed to Model: In this case the median values
        QueryPoint = pd.DataFrame([values],columns = columnLists)
        QueryPoint['item_nbr'] = itemVal
        
        
    else:
        #Final data point which will be passed to Model: In this case the exsisting value
        QueryPoint = df_chkRecord
        ActualValue = str(df_chkRecord['units'])#str(df_chkRecord.get_value(0, 'units'))
        CheckAvailable = "AVAILABLE"
    #------------------------------------------------------------    
    #Removing columns
    if 'date' in QueryPoint.columns:
        QueryPoint.drop(['date'],axis = 1,inplace=True)
    
    if 'flag' in QueryPoint.columns:
        QueryPoint.drop(['flag'],axis = 1,inplace=True)    
    #------------------------------------------------------------   
    
    chkpredict = -1
    #************************ Classification model **************
    #------------------------------------------------------------
    if len(df_chkRecord) > 0:
        lr = joblib.load('cls_LogisticRegrsn.pkl')
        prediction = lr.predict(QueryPoint)
        #print(prediction[0])
        if prediction[0] == 1:
            chkpredict = 1
        if prediction[0] == 0:
            chkpredict = 0
    #************************ Regression model ******************
    #------------------------------------------------------------
    if ((chkpredict == 1) or (len(df_chkRecord)==0)):
        if 'units' in QueryPoint.columns:
            QueryPoint.drop(['units'],axis = 1,inplace=True)
        
        #Standardize the Query point
        for clmn in QueryPoint.columns:
            # fit on data column & Transform
            scale = StandardScaler().fit(QueryPoint[[clmn]])
            QueryPoint[clmn] = scale.transform(QueryPoint[[clmn]]) 
            
        # Our best model is XGBRegressor, then Random Forest. 
        #But for both these Model- while loading pkl file we faced error in terms of Parameter so used Linear Regression   
        reg_RF = joblib.load('LinearReg.pkl')
        predictionReg = reg_RF.predict(QueryPoint)
        FinalPrediction = predictionReg
        #return jsonify({'prediction': list(predictionReg)})
    elif (chkpredict == 0) & (len(df_chkRecord)==1):
        FinalPrediction = prediction[0]
        #return jsonify({'prediction': list(prediction)})
    
    #return jsonify({'Prediction': FinalPrediction})
    #return str(FinalPrediction)
    return render_template('outputpage.html', FinalPrediction=FinalPrediction,ActualValue=ActualValue,CheckAvailable=CheckAvailable)

#
@app.route('/close', methods=['POST'])
def close():
    return render_template('close.html')


@app.route('/linkdin', methods=['POST'])
def linkdin():
    return flask.redirect('http://www.linkedin.com/in/soumenchatterje')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


