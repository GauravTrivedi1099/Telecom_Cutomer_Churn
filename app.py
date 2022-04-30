# coding: utf-8
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle


app = Flask("__name__")

@app.route('/')
def loadPage():
	return render_template('home.html', query="")


@app.route('/', methods=['POST'])
def predict():

    inputQuery1 = request.form['state']
    inputQuery2 = request.form['account_length']
    inputQuery3 = request.form['area_code']
    inputQuery4 = request.form['inter_plan']
    inputQuery5 = request.form['vm_plan']
    inputQuery6 = request.form['no_vmail_msg']
    inputQuery7 = request.form['tot_day_min']
    inputQuery8 = request.form['tot_day_calls']
    inputQuery9 = request.form['tot_day_charge']
    inputQuery10 = request.form['tot_eve_min']
    inputQuery11 = request.form['tot_eve_calls']
    inputQuery12 = request.form['tot_eve_charge']
    inputQuery13 = request.form['tot_night_min']
    inputQuery14 = request.form['tot_night_calls']
    inputQuery15 = request.form['tot_night_charge']
    inputQuery16 = request.form['tot_intl_min']
    inputQuery17 = request.form['tot_intl_calls']
    inputQuery18 = request.form['tot_intl_charge']
    inputQuery19 = request.form['cus_ser_calls']
    
    model = pickle.load(open("model.pkl", "rb"))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11,inputQuery12, inputQuery13, inputQuery14, 
             inputQuery15,inputQuery16,inputQuery17,inputQuery18,inputQuery19]]
    
    new_df = pd.DataFrame(data, columns = [inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
                                           inputQuery8, inputQuery9, inputQuery10, inputQuery11,inputQuery12, inputQuery13, inputQuery14, 
                                           inputQuery15,inputQuery16,inputQuery17,inputQuery18,inputQuery19])


    result = model.predict(new_df.tail(1))

    if result[0]==1:
        out = "Customer will churn"
    else:
        out = "Customer will not churn"
    

        
    return render_template('home.html',
                           state = request.form['state'], 
                           account_length = request.form['account_length'],
                           area_code = request.form['area_code'],
                           inter_plan = request.form['inter_plan'],
                           vm_plan = request.form['vm_plan'], 
                           no_vmail_msg = request.form['no_vmail_msg'], 
                           tot_day_min = request.form['tot_day_min'], 
                           tot_day_calls = request.form['tot_day_calls'], 
                           tot_day_charge = request.form['tot_day_charge'], 
                           tot_eve_min = request.form['tot_eve_min'], 
                           tot_eve_calls = request.form['tot_eve_calls'],
                           tot_eve_charge = request.form['tot_eve_charge'],
                           tot_night_min = request.form['tot_night_min'],
                           tot_night_calls = request.form['tot_night_calls'],
                           tot_night_charge = request.form['tot_night_charge'],
                           tot_intl_min = request.form['tot_intl_min'],
                           tot_intl_calls = request.form['tot_intl_calls'],
                           tot_intl_charge = request.form['tot_intl_charge'],
                           cus_ser_calls = request.form['cus_ser_calls']
                           ,output=out)
    

if __name__ == '__main__':
    app.run()

