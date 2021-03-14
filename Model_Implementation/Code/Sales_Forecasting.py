# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 04:00:25 2019

@author: Alekhya Bhupati
"""


#%% Import libaries
import pandas as pd
import numpy as np
import json
import copy
import os
import math
from math import sqrt
import matplotlib.pyplot as plt
import statistics
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint as sp_randint
from IPython import get_ipython

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic("matplotlib", "inline")


#%%
def user_input():
    forecast_period = input("What is the Future Forecast period? ")
    forecast_period = int(forecast_period)

    input_data = input("Enter dataset: ")


    file_type = os.path.splitext(input_data)[1]
    if file_type == '.csv':
        dataset = pd.read_csv(input_data)

    elif file_type == '.json':
        dataset = retrieve_data(input_data)

    return forecast_period, dataset

def retrieve_data(path):
    datasets = dict()
    with open(path) as json_file:
        raw_data = json.load(json_file)

        for element in raw_data:
            df_name = element['sku']
            data = pd.DataFrame()
            data['time'] = element['time']
            data['sales'] = element['sales']
            data['sales'] = [np.nan if pd.isnull(i) else int(i) for i in data['sales']]

            data = data.T
            data = data.rename(columns = data.iloc[0]).drop(data.index[0])
            datasets[df_name] = copy.deepcopy(data)

    return datasets


def dickeyfullertest(series):
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index= ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    if dfoutput['p-value'] > 0.05:
        return 0 #not stationary

    else:
        return 1 #stationary

def mean_standard_deviation(dataset):
    mu = np.mean(dataset.values)
    sd = np.std(dataset.values)

    ub = mu + (3 * sd)
    lb = mu - (3* sd)

    return lb,ub

def acf_plot1(dataset,freq):
#    plot_acf(dataset)
    res = acf(dataset)
    ub=1.96/np.sqrt(len(dataset))
    for i in range(1, len(res)-1):
        if(res[i] > ub and res[i + 1] < ub):
            p = i
            if (p > freq):
                p = freq
            break
    else:
        p = freq
    return p

def acf_plot(dataset,freq):
    res = acf(dataset)
    plot_acf(dataset)
    acfval=[0]*len(res)
    ub=1.96/np.sqrt(len(dataset))
    p1=1
    for i in range(len(res)):
        acfval[i]=abs(res[i])
    acfval.sort(reverse=True)
    acfval=np.array(acfval[1:])
    pshort=np.array(acfval[0:3])
    pshortind=[0]*len(pshort)
#    print(pshort)
    for i in range(len(pshort)):
        pshortind[i]=np.where(abs(res)==pshort[i])[0][0]
    ind=np.where(acfval>ub)[0]
    finalacf=acfval[ind]
    plist=[0]*len(finalacf)
    for i in range(len(finalacf)):
        plist[i]=np.where(abs(res)==finalacf[i])[0][0]
#    return pshortind,plist
#    print(plist)
    while len(finalacf)>0:
        p1=np.where(abs(res)==max(finalacf))[0][0]
        if p1 > freq:
            finalacf=finalacf[1:]
        else:
            break
    return p1,pshortind,plist

def pacf_plot1(dataset,freq):
#    plot_pacf(dataset)
    res = pacf(dataset)
    ub=1.96/np.sqrt(len(dataset))
    for i in range(1, len(res)-1):
        if(res[i] > ub and res[i + 1] < ub):
            q = i
            if (q > freq/2):
                q = freq/2
            break
    else:
        q = freq/2
    return int(q)

def pacf_plot(dataset,freq):
    res = pacf(dataset)
    plot_pacf(dataset)
    pacfval=[0]*len(res)
    ub=1.96/np.sqrt(len(dataset))
    q1=0
    for i in range(len(res)):
        pacfval[i]=abs(res[i])
    pacfval.sort(reverse=True)
    pacfval=np.array(pacfval[1:])
    ind=np.where(pacfval>ub)[0]
    finalpacf=pacfval[ind]
    while len(finalpacf)>0:
        q1=np.where(abs(res)==max(finalpacf))[0]
        q1=q1[0]
        if q1 > int(freq/2):
            finalpacf=finalpacf[1:]
        else:
            break
    return q1

#TODO: Send alpha, beta
def median_absolute_deviation(dataset, median):
    median_list = list()
    dataset.reset_index(drop = True, inplace = True)
    for i in range(0, len(dataset)):
        value = dataset.T[i] - median
        median_list.append(value)
    ms = np.abs(median_list)
    mad = np.median(ms)
    ub = median + (3 * mad)
    lb = median - (3 * mad)


    return ub,lb

def dateformat(dataset):
    dataset['time'] = dataset.index.tolist()
    dataset['time']=pd.to_datetime(dataset['time'], format='%m/%d/%y',infer_datetime_format=True)
    dataset.set_index('time', inplace = True)
    return dataset

def outlier_treatment(dataset):
    median = np.median(dataset)
    if median == 0:
        ub,lb = mean_standard_deviation(dataset)
    else:
        ub,lb = median_absolute_deviation(dataset, median)
    new_dataset = np.clip(dataset, lb, ub)
    return new_dataset

#pt--->outlier bucket size
#sindex ---> number of zeros to categorize as sparse data
#freq ---> seasonality
def get_bucket_size(interval):
    interval_type = find_interval_type(interval) #aggregation (weekly/monthly)
    if interval_type == 'W':
        pt=12
        sindex=24
        freq=52
    elif interval_type=='M'or interval_type=='Random':
        pt=6
        sindex=10
        freq=12
    elif interval_type=='Y':
        pt=2
        sindex=0
        freq=0
    return pt,sindex,freq

def outlier_treatment_tech(dataset,interval,pt): 
    start=0
    end=pt
    sku_data=[0]*len(dataset)
    while end < len(dataset):
        sku_data[start:end]=outlier_treatment(dataset[start:end])
        start=end
        end+=pt
    if start < len(dataset):
        sku_data[start:len(dataset)]=outlier_treatment(dataset[start:end])
    sku_data=pd.DataFrame(sku_data)
    return sku_data

def Sesonal_detection(sku_data):
    median = np.median(sku_data)

    if median == 0:
        ub,lb = mean_standard_deviation(sku_data)
    else:
        ub,lb = median_absolute_deviation(sku_data, median)
    outliers1= sku_data > ub
    outliers2 = sku_data < lb
    a=np.where(outliers1==True)[0]
    b=np.where(outliers2==True)[0]
    flag1=flag2=1
    if len(a)==0:
        flag1=0
        remove1=[]
    if len(b)==0:
        flag2=0
        remove2=[]

    if flag1==1:
        k=np.zeros([len(a)-1, len(a)])
        for i in range(0,(len(a)-1)):
            for j in range(1,len(a)):
                if a[j]==(a[i]+12) or a[j]==(a[i]+24):
                    k[i][j]=1
                else:
                    k[i][j]=0
        m=np.where(k!=0)
        z=np.unique(m).tolist()
        remove1=a[z]
    if flag2==1:
        q=np.zeros([len(b)-1, len(b)])
        for i in range(0,(len(b)-1)):
            for j in range(1,len(b)):
                if b[j]==(b[i]+12) or b[j]==(b[i]+24):
                    q[i][j]=1
                else:
                    q[i][j]=0
        n=np.where(q!=0)
        z1=np.unique(n).tolist()
        remove2=b[z1]
    return remove1,remove2,flag1,flag2

def impute_missing_dates(dataset):
    interval,start_date,end_date=find_interval(dataset.index)
    drange = pd.date_range(start_date, end_date, freq = interval)
    comp_data = pd.DataFrame(drange, columns = ['time'])
    sales=dataset['sales'].to_dict()
    comp_data['sales'] = comp_data['time'].map(sales)
    comp_data.drop('time', axis = 1, inplace = True)
    return comp_data, interval

def find_interval(date):
    date=pd.to_datetime(date, format='%m/%d/%Y',infer_datetime_format=True)
    diff=[]
    for i in range(len(date)-1):
        interval=date[i+1]-date[i]
        diff.append(interval)
    mode=statistics.mode(diff)

    return mode,date[0],date[-1]


def find_interval_type(interval):
    interval=interval.days
    if interval==7:
        itype='W'
    elif interval==30 or interval==31:
        itype='M'
    elif interval==365:
        itype='Y'
    else:
        itype='Random'

    return itype

def data_imputation_zero(dataset):
    dataset.fillna(0, inplace = True)
    return dataset

#TODO: Send as transpose of dataset
def data_imputation(dataset,freq):
    #Taking the mean of nearest neighbours to fill NA

    data_forward = dataset.fillna(method = 'ffill')
    data_back = dataset.fillna(method = 'bfill')
    data_back.fillna(0, inplace = True)
    data_forward.fillna(0, inplace = True)

    new_data = (data_forward.values + data_back.values) / 2
    dataset=pd.DataFrame(dataset.values)

    imput=dataset.isnull()
    imput=imput[0]

    dataset=dataset[0]
    for i in range(len(dataset)):
        div_factor = 3
        if imput[i]==True:
            #Negative index, set previous as 0
            if i - freq < 0:
                prev_value = 0
                div_factor -= 1
            #Fetch previous value
            else:
                prev_value = dataset[i - freq]

            #Outside boundary or next value is NaN, set previous as 0
            if i + freq >= len(dataset) or imput[i + freq]==True:
                next_value = 0
                div_factor -= 1
            else:
                next_value = dataset[i + freq]
            dataset[i] = (new_data[i] + prev_value + next_value)/div_factor

    df = pd.DataFrame(dataset)
    return df

#reading from first non-zero
def read_from_first_sales(sku_data):
    test=pd.isnull(sku_data)
    index=np.where(test==False)[0]
    index=index[0]
    sku_data = sku_data[index:]
    sku_data=sku_data.reset_index(drop = True)
    return sku_data

def new_forecast(data,forecast,forecast_period):

    sample_data=data[-forecast_period:]
    sample_data=pd.DataFrame(sample_data)
    forecast=pd.DataFrame(forecast)
    median_data=np.median(sample_data)
    median_fore=np.median(forecast)
    if median_data == 0:
        ub_d,lb_d = mean_standard_deviation(sample_data)
    else:
        ub_d,lb_d = median_absolute_deviation(sample_data, median_data)
    if median_fore == 0:
        ub_f,lb_f = mean_standard_deviation(forecast)
    else:
        ub_f,lb_f = median_absolute_deviation(forecast, median_fore)
    if ub_f >ub_d:
        ub=ub_d
    else:
        ub=ub_f
    if lb_f >lb_d:
        lb=lb_d
    else:
        lb=lb_f
    forecast=np.clip(forecast,lb,ub)
    return forecast



'''
Function used to fit the model
'''

def fit_model(train_data, model):

    X, y = train_data[:, 0:-1], train_data[:, -1]
    model.fit(X, y)
    return model


'''
Function used for one-step forecasting
'''

def forecast_model(model, X):

    yhat = model.predict(X)

    return yhat


'''
Function used for plotting

'''
def plotting(key, predictions, expected):
    rmse = calculate_rmse(key, expected, predictions)
    plt.figure()
    plt.title(key)

    y_values = list(expected) + list(predictions)
    y_range = max(y_values) - min(y_values)
    plt.text(6, min(y_values) + 0.2 * y_range, 'RMSE = ' + str(rmse))
    plt.plot(predictions)
    plt.plot(expected)
    plt.legend(['predicted', 'expected'])
    plt.show()


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = sum(np.abs((y_true - y_pred) / y_true)* 100)/len(y_true)
    if np.isnan(mape)== True or np.isfinite(mape)==False:
        mape=0
    return mape


def calculate_facc(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    facc = 1 - (sum(np.abs(y_true - y_pred)) / sum(y_true))
    if np.isnan(facc)== True or np.isfinite(facc)==False:
        facc=0
    return facc * 100



def calculate_rmse(key, expected, predictions):
    expected=np.array(expected)
    rmse = sqrt(mean_squared_error(expected, predictions))
    print("RMSE FOR %s: %d " % (key, rmse))
    return rmse


def moving_average(test,n,n1):
    train = []
    train = [x for x in test]
    pred = []
    for num in range(n):
        test_new = pd.DataFrame(train)
        pred1 = (test_new.tail(n1).mean())
        pred1 = pred1[0]
        pred.append(pred1)
        train.append(pred1)

    return pred

def weighted_moving_average(test1,n,n1):
    alpha=[0.25,0.3,0.45]
    train = [x for x in test1]
    pred = []
    for num in range(n):
        test_new = pd.DataFrame(train)
        pred1 = test_new.tail(n1)
        pred1=np.dot(pred1[0],alpha)
        pred.append(pred1)
        train.append(pred1)
    pred = [int(i) for i in pred]
    return pred

def check_repetition(arr, limit, index_start, index_end):
    length = index_start
    try:
        for i in range(0, int( len(arr)/length)):
            condition  = np.array( arr[i:int(i+length)]) - np.array( arr[int( i+length):int(i+2*length)])
            condition  = np.sum([abs(number) for number in condition])
            if condition >= limit :
                if length + 1 <= index_end:
                    return check_repetition(arr, limit, length + 1, index_end)
            # if not than no more computations needed
                else:
                    return 0

            if i == int( len(arr)/length)-2:
                return(length)
    except:
        for i in range(0, int( len(arr)/length)):
            if  i+2*length+1 <= index_end and i+length+1 <= index_end:
                break
            condition  = np.array( arr[i:int(i+length)]) - np.array( arr[int( i+length):int(i+2*length)])
            condition  = np.sum([abs(number) for number in condition])
            if condition >= limit :
                if length + 1 <= index_end:
                    return check_repetition(arr, limit, length + 1, index_end)
            # if not than no more computations needed
                else:
                    return 0

            if i == int( len(arr)/length)-2:
                return(length)

    return 0

def Croston(dataset,forecast_period=1,alpha=0.4):
    d = np.array(dataset) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d, [np.nan] * forecast_period) # Append np.nan into the demand array to cover future periods

    #level (a), periodicity(p) and forecast (f)
    a, p, f = np.full((3, cols + forecast_period), np.nan)
    q = 1 #periods since last demand observation

    # Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0] / p[0] # Create all the t+1 forecasts
    for t in range(0,cols):
        if d[t] > 0:
            a[t +1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = alpha * q + (1 - alpha) * p[t]
            f[t + 1] = a[t + 1] / p[t + 1]
            q = 1
        else:
            a[t + 1] = a[t]
            p[t + 1] = p[t]
            f[t + 1] = f[t]
            q += 1

    # Future Forecast
    a[cols + 1 : cols + forecast_period] = a[cols]
    p[cols + 1 : cols + forecast_period] = p[cols]
    f[cols + 1 : cols + forecast_period] = f[cols]

    rmse = calculate_rmse('Croston', d[0:cols - 1], f[0:cols - 1])
#    plotting('Croston', f[0 : cols - 1], d[0 : cols - 1])
    forecast = f[cols : cols + forecast_period]

    return rmse,forecast


def Croston_TSB(dataset, forecast_period = 1, alpha = 0.4, beta = 0.4):
    rmse=dict()
    pred_croston=dict()
    rmse_val=[]
    d = np.array(dataset) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d, [np.nan] * forecast_period) # Append np.nan into the demand array to cover future periods

    #level (a), probability(p) and forecast (f)
    a, p, f = np.full((3, cols + forecast_period), np.nan)# Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1/(1 + first_occurence)
    f[0] = p[0] * a[0]

    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = beta * (1) + (1 - beta) * p[t]
        else:
            a[t + 1] = a[t]
            p[t + 1] = (1 - beta) * p[t]
        f[t + 1] = p[t + 1] * a[t + 1]

    # Future Forecast
    a[cols + 1 : cols + forecast_period] = a[cols]
    p[cols + 1 : cols + forecast_period] = p[cols]
    f[cols + 1 : cols + forecast_period] = f[cols]

    rmse_val.append(calculate_rmse('Croston', d[0:cols - 1], f[0:cols - 1]))
    rmse['Croston']=rmse_val
    plotting('Croston', f[0 : cols - 1], d[0 : cols - 1])
    forecast = f[cols : cols + forecast_period]
    forecast = [int(i) for i in forecast]
    pred_croston['Croston']=forecast
    return rmse, pred_croston


'''
Train the models for each SKU and forecast the predictions

'''
def training(datasets, forecast_period):
    forecast_results = []
    num = 0
    for incr,sku in enumerate(datasets):
        num += 1

        print("------------------------------------------------------------")
        print("Running SKU %d: %s..." % (num, sku))


        raw_data = copy.deepcopy(datasets[sku].T)
        output = init_output(forecast_period, raw_data)
        dataset = raw_data.copy()

        dataset = dataset[:-1]
        if((dataset['sales'] == 0).all() == True or (set([math.isnan(x) for x in dataset['sales']]) == {True})):
            print(dataset['sales'])
            print("All zeros/NaNs")
            forecast = [0] * forecast_period
            output['forecast_values'] = assign_dates(forecast, 'forecast', dataset.tail(1))
            output['facc'], output['mape'], output['bias'] = calculate_forecast_accuracy(raw_data.iloc[-1], forecast[0])

            forecast_results = output_forecast(sku, dataset, datasets[sku].T, output, forecast_results)
            continue

        sku_data = dataset.astype(np.float32)
        sku_data = read_from_first_sales(sku_data['sales'])

#size--->outlier bucket size
#sparse_size ---> number of zeros to categorize as sparse data
#freq ---> seasonality
        interval=30
        size=6
        freq=12
        test_nan=pd.DataFrame(sku_data[-freq:])
        test_nan=test_nan['sales']

#if last 1 year is NaN, impute data with zero and forecast is MA(6)

        if sum(test_nan.isnull())>=freq:
            print("Last 1 year NaN")
            sku_data = data_imputation_zero(test_nan)
            forecast = moving_average(sku_data,forecast_period,6)
            output['forecast_values'] = assign_dates(forecast, 'forecast', dataset.tail(1))
            output['facc'], output['mape'], output['bias'] = calculate_forecast_accuracy(raw_data.iloc[-1], forecast[0])

            forecast_results = output_forecast(sku, dataset, sku_data, output, forecast_results)
            continue

#if # NaNs more than 60% impute with 0 else impute with values

        if sum(pd.isnull(sku_data))>(0.6*len(sku_data)):
            print("Nan Greater than 60%")
            sku_data = data_imputation_zero(sku_data)

        else:
            print("Nan less than 60%")
            sku_data = data_imputation(sku_data,freq)
            sku_data=sku_data[0]

        sku_data = read_from_first_sales(sku_data)

#After reading from first non-zero if data is insufficient ---> weighted MA(3)

        if len(sku_data) < 20:
            try:
                print("Weighted Moving Average")
                forecast = weighted_moving_average(sku_data,forecast_period,3)
                output['forecast_values'] = assign_dates(forecast, 'forecast', dataset.tail(1))
                output['facc'], output['mape'], output['bias'] = calculate_forecast_accuracy(raw_data.iloc[-1], forecast[0])

                forecast_results = output_forecast(sku, dataset, sku_data, output, forecast_results)
            except:
                print("Less than 3")
                forecast = moving_average(sku_data, forecast_period, len(sku_data))
                output['forecast_values'] = assign_dates(forecast, 'forecast', dataset.tail(1))
                output['facc'], output['mape'], output['bias'] = calculate_forecast_accuracy(raw_data.iloc[-1], forecast[0])

                forecast_results = output_forecast(sku, dataset, sku_data, output, forecast_results)

            continue

        data_copy=sku_data.copy()
        data_copy=np.array(data_copy)

        index1,index2,sflag1,sflag2=Sesonal_detection(sku_data)
        sku_data = outlier_treatment_tech(sku_data,interval,size)
        sku_data=np.array(sku_data[0])


        if sflag1==1:
            sku_data[index1]=data_copy[index1]
        if sflag2==1:
            sku_data[index2]=data_copy[index2]
        else:
            sku_data=sku_data

        sku_data=pd.DataFrame(sku_data)

        #Testing Stationarity
        d = 0
        df_test_result = dickeyfullertest(sku_data.T.squeeze()) #pd.Series(sku_data[0])

        while df_test_result == 0:
            d += 1
            if d == 1:
                new_data = difference(sku_data[0].tolist())
            else:
                new_data = difference(new_data)
            df_test_result = dickeyfullertest(new_data)
        sample=np.array(sku_data)
        repeat=check_repetition(sample, freq , 1, len(sample))
        #Finding p and q value
        try:
            if d == 0:
                p1,ps,pl = acf_plot(sku_data,freq)
                q = pacf_plot(sku_data,freq)
                data = sku_data
            else:

                p,ps,pl = acf_plot(new_data,freq)
                q = pacf_plot(new_data,freq)
                data = new_data

            if repeat in ps:
                p=repeat
            elif repeat in pl:
                p=repeat
            else:
                p=pl[0]
            if p > freq:
                p=freq
        except:
            p=1
            q=1
            data=sku_data

        data=sku_data
        best_order = (p, d, q)
        print("BEST ORDER :",best_order)
        tsize=5
        expected = data[-tsize:].reset_index(drop = True)
        expected = [float(i) for i in expected.values]
        train_6wa=sku_data[0:-tsize]
        predictions_ML,rmse_ML = time_series_using_ml(sku_data, tsize, best_order)
        rmse_ARIMA, rmse_ES, rmse_naive,rmse_ma, predictions_ARIMA, predictions_ES , predictions_naive, predictions_ma = time_series_models(freq,sku_data, data, tsize, best_order)
        print("Modeling done")

        rmse_TS = rmse_ARIMA.copy()
        rmse_TS.update(rmse_ES)
        rmse_TS.update(rmse_naive)
        rmse_TS.update(rmse_ma)

        predictions = predictions_ML
        predictions.update(predictions_ARIMA)
        predictions.update(predictions_ES)
        predictions.update(predictions_naive)
        predictions.update(predictions_ma)


        rmse_Croston,predictions_Croston=Croston_TSB(sku_data,tsize)
        rmse_TS.update(rmse_Croston)
        predictions.update(predictions_Croston)

        rmse_vol_ml=dict()
        for key in rmse_ML:
            mean = np.mean(rmse_ML[key])
            rmse_vol_ml[key]= mean

        rmse_vol_ts=dict()
        for key in rmse_TS:
            mean = np.mean(rmse_TS[key])
            rmse_vol_ts[key] = mean


        #Top 3 models
        best_models_ml =  sorted(rmse_vol_ml, key=rmse_vol_ml.get, reverse=False)[:3]
        best_models_ts =  sorted(rmse_vol_ts, key=rmse_vol_ts.get, reverse=False)[:3]
        bias_ml = []
        accuracy_ml = []
        for model in best_models_ml:
            bias_ml.append((sum(expected) - sum(predictions[model]))/len(expected))
            accuracy_ml.append(calculate_facc(expected, predictions[model]))
        bias_ml=[float(format(i,'.3f')) for i in bias_ml]
        accuracy_ml=[float(format(i,'.3f')) for i in accuracy_ml]

        bias_ts = []
        accuracy_ts = []
        for model in best_models_ts:
            bias_ts.append((sum(expected) - sum(predictions[model]))/len(expected))
            accuracy_ts.append(calculate_facc(expected, predictions[model]))
        bias_ts=[float(format(i,'.3f')) for i in bias_ts]
        accuracy_ts=[float(format(i,'.3f')) for i in accuracy_ts]

        #For one ensemble
        error_ml = min(rmse_vol_ml.values())
        error_ts = min(rmse_vol_ts.values())


        best_models = [min(rmse_vol_ml, key = lambda x : rmse_vol_ml.get(x)), min(rmse_vol_ts, key = lambda x : rmse_vol_ts.get(x))]
        print("BEST MODELS :",best_models)
        print("ERRORS OF BEST MODELS :",error_ml, error_ts)
        forecast_ml = model_predict(best_models[0], best_order,data, forecast_period)
        if best_models[1]=='Croston':
            rmse_Croston,forecast_ts=Croston_TSB(sku_data,forecast_period)
            forecast_ts=forecast_ts['Croston']
        else:
            forecast_ts = model_predict(best_models[1], best_order, sku_data, forecast_period,repeat)

        forecast_ml = [0 if i < 0 else int(i) for i in forecast_ml]
        forecast_ts = [0 if i < 0 else int(i) for i in forecast_ts]

        weight_ts,weight_ml=weight_calculation(data,best_models,best_order)
        print("weight ts:" ,weight_ts)
        print("weight ml:" ,weight_ml)

        Vm=predictions[best_models[0]]
        Vt=predictions[best_models[1]]

        Ve=method_ensemble(Vm,Vt,weight_ml,weight_ts,tsize)
        error_en=calculate_rmse('Ensemble', expected, Ve)

        bias_en=[]
        accuracy_en=[]

        bias_en.append((sum(expected) - sum(Ve))/len(expected))
        accuracy_en.append(calculate_facc(expected, Ve))
        bias_en=[float(format(i,'.3f')) for i in bias_en]
        accuracy_en=[float(format(i,'.3f')) for i in accuracy_en]
        #Ensemble of six month naive and weighted average
        V6wa,rmse_6wa = model_Naive('naive6wa',train_6wa, tsize, (0,0,0), 0 , train_flag = 1)
        error_6wa=np.mean(rmse_6wa)
        forecast_6wa = model_predict('naive6wa', best_order,data, forecast_period)

        forecast_en=method_ensemble(forecast_ml,forecast_ts,weight_ml,weight_ts,forecast_period)

        output['forecast_period'] = forecast_period
        output['interval'] = 'M'
        output['best_models_ml'] = best_models_ml
        output['best_models_ts'] = best_models_ts
        output['bias_ml'] = bias_ml
        output['bias_ts'] = bias_ts
        output['bias_en'] = bias_en
        output['accuracy_ml'] = accuracy_ml
        output['accuracy_ts'] = accuracy_ts
        output['accuracy_en'] = accuracy_en

        error_min_model=min(error_ml,error_ts,error_en)


        print("Errors:", )
        print("ML:", error_ml)
        print("TS:", error_ts)
        print("Ensemble:", error_en)
        print("six_naive_WA",error_6wa)

        min_error=min(error_min_model,error_6wa)

        if min_error==error_6wa or all(elem == forecast_ts[0] for elem in forecast_ts)==True or all(elem == forecast_ml[0] for elem in forecast_ml)==True or all(elem == forecast_en[0] for elem in forecast_en)==True:
            print("Best forecast from six naive")
            forecast = forecast_6wa
            output['validation'] = assign_dates(V6wa, 'validation', dataset.tail(5))
            validation_facc = calculate_validation_facc(expected, V6wa)
            output['validation_facc'] = assign_dates(validation_facc, 'val_facc', dataset.tail(5))
        elif min_error==error_ml:
            print("Best forecast from ML")
            forecast = forecast_ml
            output['validation'] = assign_dates(Vm, 'validation', dataset.tail(5))
            validation_facc = calculate_validation_facc(expected, Vm)
            output['validation_facc'] = assign_dates(validation_facc, 'val_facc', dataset.tail(5))
        elif min_error==error_en:
            print("Best forecast from Ensemble")
            forecast=forecast_en
            output['validation'] = assign_dates(Ve, 'validation', dataset.tail(5))
            validation_facc = calculate_validation_facc(expected,Ve)
            output['validation_facc'] = assign_dates(validation_facc, 'val_facc', dataset.tail(5))
        elif min_error==error_ts:
            print("Best forecast from TS")
            forecast=forecast_ts
            output['validation'] = assign_dates(Vt, 'validation', dataset.tail(5))
            validation_facc = calculate_validation_facc(expected, Vt)
            output['validation_facc'] = assign_dates(validation_facc, 'val_facc', dataset.tail(5))
#
        print("Forecasts:")
        print("ML:", forecast_ml)
        print("TS:", forecast_ts)
        print("Ensemble:", forecast_en)
        print("Best Forecast",forecast)

        output['forecast_values'] = assign_dates(forecast, 'forecast', dataset.tail(1))
        output['facc'], output['mape'], output['bias'] = calculate_forecast_accuracy(raw_data.iloc[-1].sales, forecast[0])


        output['forecast_ml'] = forecast_ml
        output['forecast_ts'] = forecast_ts
        output['forecast_en'] = forecast_en
        output['error_ml'] = error_ml
        output['error_ts'] = error_ts
        output['error_en'] = error_en
        output['weight_ts'] = weight_ts
        output['weight_ml'] = weight_ml
        output['model_ml'] = best_models[0]
        output['model_ts'] = best_models[1]


        forecast_results = output_forecast(sku, dataset, sku_data, output, forecast_results)


        plot_all_forecasts(dataset, sku_data, forecast,forecast_en,forecast_ml, forecast_ts, sku)

    return forecast_results

'''
Function to evaluate an ARIMA model for a given order (p, d, q)

'''

def evaluate_arima_model(X, arima_order):
    #Prepare training dataset
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]

    #Evaluate ARIMA model
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit(disp = 0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(yhat)

    #Measure error using RMSE
    error = sqrt(mean_squared_error(test, predictions))
    return error

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)

                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order

                except:
                    continue

    return best_cfg

def model_ARMA(key,train, test_shape, order, train_flag = 0, test = []):
    predictions = []
    rmse_val=[]
    if(train_flag):
        test=test[0]
    try:
        train = train.values
    except:
        train = train
    history = [np.asscalar(x) for x in train]

    if train_flag==1:
        itr=5
        data=pd.DataFrame(history)
        for i in range(3):
            pred_temp = []
            v_train=[np.asscalar(x) for x in data[:-itr].values]
            v_expected=data.tail(itr).head(3).reset_index(drop = True)
            try:
                for j in range(3):
                    model = ARMA(v_train, order = order)
                    model_fit = model.fit(disp=0, transparams=False, trend='nc')
                    yhat = model_fit.forecast()[0]
                    pred = yhat
                    if pred < 0:
                        pred = weighted_moving_average(v_train, 1, 3)
                        pred = pred[0]
                    pred_temp.append(pred)
                    v_train.append(pred)

            except:
                pred_temp.extend(moving_average(v_train, 3 - len(pred_temp), 3))
            plotting(key, pred_temp, v_expected)

            if i == 2:
                predictions.extend(pred_temp)
            else:
                predictions.append(pred_temp[0])

            rmse_val.append(calculate_rmse(key, v_expected, pred_temp))
            itr=itr-1

    else:
        try:
            for t in range(test_shape):
                model = ARMA(history, order = order)
                model_fit = model.fit(disp=0, transparams=False, trend='nc')
                yhat = model_fit.forecast()[0]
                inverted = list()
                for i in range(len(yhat)):
                    value = inverse_difference(history, yhat[i], len(history) - i)
                    inverted.append(value)
                inverted = np.array(inverted)
                pred = inverted[-1]
                if pred < 0:
                    pred = weighted_moving_average(history, 1, 3)
                predictions.append(pred)
                history.append(yhat)
        except:
            predictions.extend(moving_average(history, test_shape - len(predictions), 3))

    predictions = [int(i) for i in predictions]
    return predictions,rmse_val

def model_ARIMA(key,train, test_shape, order, train_flag = 0, test = []):
    predictions = []
    rmse_val = []
    if(train_flag):
        test=test[0]
    try:
        train = train.values
    except:
        train = train
    history = [np.asscalar(x) for x in train]

    if train_flag==1:
        itr=5
        data=pd.DataFrame(history)
        for i in range(3):
            pred_temp = []
            v_train=[np.asscalar(x) for x in data[:-itr].values]
            v_expected=data.tail(itr).head(3).reset_index(drop = True)
            try:
                order=(order[0],1,order[2])
                for j in range(3):
                    model = ARIMA(v_train, order = order)
                    model_fit = model.fit(disp=0)
                    yhat = model_fit.forecast()[0]
                    if yhat < 0:
                        yhat= weighted_moving_average(history,1,3)
                        yhat=yhat[0]
                    pred_temp.append(yhat)
                    v_train.append(yhat)
            except:
                pred_temp.extend(moving_average(v_train, 3 - len(pred_temp), 3))
            plotting(key, pred_temp, v_expected)
            rmse_val.append(calculate_rmse(key, v_expected, pred_temp))

            if i == 2:
                predictions.extend(pred_temp)
            else:
                predictions.append(pred_temp[0])

            itr=itr-1
    else:
        try:
            #TODO:check order
            order=(order[0],1,order[2])
            for t in range(test_shape):
                model = ARIMA(history, order = order)
                model_fit = model.fit(disp=0)
                yhat = model_fit.forecast()[0]
                if yhat < 0:
                    yhat= weighted_moving_average(history,1,3)
                yhat=yhat[0]
                predictions.append(yhat)
                history.append(yhat)
        except:
                predictions.extend(moving_average(history, test_shape - len(predictions), 3))

    predictions = [0 if pd.isnull(i) else int(i) for i in predictions]
    return predictions,rmse_val

def model_ES(key, train, test_shape = 0, train_flag = 0, test = []):
    predictions = []
    rmse_val=[]

    try:
        train = train.values
    except:
        train = train
    history = [np.asscalar(x) for x in train]

#   TRAIN
    if train_flag==1:
        itr=5
        data=pd.DataFrame(history)
        for i in range(3):
            pred_temp = []
            v_train=[np.asscalar(x) for x in data[:-itr].values]
            v_expected=data.tail(itr).head(3).reset_index(drop = True)
            try:
                for t in range(3):
                    if key=='SES':
                        model = SimpleExpSmoothing(history)
                    elif key=='HWES':
                         model = ExponentialSmoothing(history)
                    model_fit = model.fit()
                    yhat= model_fit.predict(len(history), len(history))
                    if yhat < 0:
                        yhat= weighted_moving_average(history,1,3)
                    yhat=yhat[0]
                    pred_temp.append(yhat)
                    v_train.append(yhat)
            except:
                pred_temp.extend(moving_average(v_train, 3 - len(pred_temp), 3))
            plotting(key, pred_temp, v_expected)
            rmse_val.append(calculate_rmse(key, v_expected, pred_temp))

            if i == 2:
                predictions.extend(pred_temp)
            else:
                predictions.append(pred_temp[0])

            itr=itr-1
#   FORECAST
    else:
        try:
            for t in range(test_shape):
                if key=='SES':
                    model = SimpleExpSmoothing(history)
                elif key=='HWES':
                     model = ExponentialSmoothing(history)
                model_fit = model.fit()
                yhat= model_fit.predict(len(history), len(history))
                if yhat < 0:
                    yhat= weighted_moving_average(history,1,3)
                yhat=yhat[0]
                predictions.append(yhat)
                history.append(yhat)
        except:
            predictions.extend(moving_average(history, test_shape - len(predictions), 3))

    predictions = [int(i) for i in predictions]
    return predictions,rmse_val

def model_Naive(key,train,test_shape,order,rept,train_flag = 0):
   forecast = []
   p=order[0]
   rmse_val=[]
   try:
       train = train.values
   except:
       train = train

   history = [np.asscalar(x) for x in train]
   if train_flag==1:
        itr=5
        data=pd.DataFrame(history)
        for i in range(3):
            pred_temp = []
            v_train=data[:-itr]
            v_train=v_train.values
            v_expected=data.tail(itr).head(3)

            for j in range(3):

                if key =='naive':
                    try:
                        t = v_train[-p]
                    except:
                        t=0

                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key == 'naive_rept':
                    try:
                        t = v_train[-rept]
                    except:
                        t=0
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key == 'naive3':
                    try:
                        t = v_train[-3]
                    except:
                        t=0
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key == 'naive6':
                    try:
                        t = v_train[-6]
                    except:
                        t=0
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key == 'naive12':
                    try:
                        t=v_train[-12]
                    except:
                        t=0
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key == 'naive12wa':
                    try:
                        yt=v_train[-12]
                    except:
                        yt=0
                    try:
                        yt_1=v_train[-24]
                    except:
                        yt_1=0
                    t = ((0.55*yt)+(0.45*yt_1))
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key=='naive6wa':
                    try:
                        #naive of six
                        try:
                            naive_six=v_train[-6]
                        except:
                            naive_six=0
                        #weighted moving average
                        alpha=[0.25,0.35,0.4]
                        pred1 = v_train[-3:]
                        pred1=[np.asscalar(x) for x in pred1]
                        weighted_avg=np.dot(pred1,alpha)
                        #ensemble
                        t=(0.75*naive_six)+(0.25*weighted_avg)
                    except:
                        t=0
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)


            rmse_val.append(calculate_rmse(key, v_expected, pred_temp))

            if i == 2:
                forecast.extend(pred_temp)
            else:
                forecast.append(pred_temp[0])

            itr=itr-1
   else:
        for num in range(test_shape):
           if key =='naive':
               try:
                   t = history[-p]
                   forecast.append(t)
                   history.append(t)
               except:
                   pass
           elif key == 'naive2':
               try:
                   t = history[-rept]
                   forecast.append(t)
                   history.append(t)
               except:
                   pass
           elif key == 'naive3':
               try:
                   t = history[-3]
                   forecast.append(t)
                   history.append(t)
               except:
                   pass
           elif key == 'naive6':
               try:
                   t = history[-6]
                   forecast.append(t)
                   history.append(t)
               except:
                   pass
           elif key == 'naive12':
               try:
                   t = history[-12]
                   forecast.append(t)
                   history.append(t)
               except:
                   pass
           elif key == 'naive12wa':
               try:
                   yt=history[-12]
               except:
                   yt=0
               try:
                   yt_1=history[-24]
               except:
                   yt_1=0
               t = ((0.55*yt)+(0.45*yt_1))
               forecast.append(t)
               history.append(t)
           elif key=='naive6wa':
               #naive of six
               try:
                   naive_six=history[-6]
               except:
                   naive_six=0
                #weighted moving average
               alpha=[0.25,0.35,0.4]
               pred1 = history[-3:]
               weighted_avg=np.dot(pred1,alpha)
               #ensemble
               t=(0.75*naive_six)+(0.25*weighted_avg)
               forecast.append(t)
               history.append(t)

   forecast = [int(i) for i in forecast]
   return forecast,rmse_val

def model_MA(key,train,test_shape,train_flag = 0):
    forecast = []
    rmse_val=[]
    try:
       train = train.values
    except:
       train = train
    history = [np.asscalar(x) for x in train]

# TRAIN
    if train_flag==1:
        itr=5
        data=pd.DataFrame(history)
        for i in range(3):
            pred_temp = []
            v_train=data[:-itr]
            v_train=v_train.values
            v_expected=data.tail(itr).head(3)
            for j in range(3):
                if key =='sma':
                    pred1 = np.mean(v_train[-3:])
                    pred_temp.append(pred1)
                    v_train=np.append(v_train,pred1)
                if key =='wma':
                    alpha=[0.25,0.35,0.4]
                    pred1 = v_train[-3:]
                    pred1=[np.asscalar(x) for x in pred1]
                    pred1=np.dot(pred1,alpha)
                    pred_temp.append(pred1)
                    v_train=np.append(v_train,pred1)
            rmse_val.append(calculate_rmse(key, v_expected, pred_temp))

            if i == 2:
                forecast.extend(pred_temp)
            else:
                forecast.append(pred_temp[0])

            itr=itr-1
#   FORECAST
    else:

        for num in range(test_shape):
            if key =='sma':
                test_new = pd.DataFrame(history)
                pred1 = (test_new.tail(3).mean())
                pred1 = pred1[0]
                forecast.append(pred1)
                history.append(pred1)
            if key =='wma':
                alpha=[0.25,0.35,0.4]
                test_new = pd.DataFrame(history)
                pred1 = test_new.tail(3)
                pred1=np.dot(pred1[0],alpha)
                forecast.append(pred1)
                history.append(pred1)

    forecast = [int(i) for i in forecast]
    return forecast,rmse_val


def models_ML():
    models = dict()
    n_trees = 100
    #prameters for RandomSearch
    lr_param = {"fit_intercept": [True, False],"normalize": [False],"copy_X": [True, False]}
#    knn_param = {"n_neighbors":[2,3,4,5,6,7,8],"metric": ["euclidean", "cityblock"]}
    dtree_param = {"max_depth": [3,None],"min_samples_leaf": sp_randint(1, 11),"criterion": ["mse"],"splitter": ["best","random"],"max_features": ["auto","sqrt",None]}
    lasso_param = {"alpha":[0.02, 0.024, 0.025, 0.026, 0.03],"fit_intercept": [True, False],"normalize": [True, False],"selection": ["random"]}
    ridge_param = {"alpha":[200, 230, 250,265, 270, 275, 290, 300, 500],"fit_intercept": [True, False],"normalize": [True, False],"solver": ["auto"]}
    elas_param = {"alpha": list(np.logspace(-5,2,8)),"l1_ratio": [.2,.4,.6,.8],"fit_intercept": [True, False],"normalize": [True,False],"precompute": [True, False]}

    models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
    models['lasso']             = RandomizedSearchCV(Lasso(), lasso_param, n_jobs=1, n_iter = 100,random_state=42)
    models['ridge']             = RandomizedSearchCV(Ridge(), ridge_param, n_jobs=1, n_iter = 100,random_state=42)
    models['en']                = RandomizedSearchCV(ElasticNet(), elas_param,scoring='neg_mean_squared_error', n_jobs=1, n_iter= 100,cv=10,random_state=42)
    models['huber']             = HuberRegressor()
    models['llars']             = LassoLars()
    models['pa']                = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3,random_state=42)
    models['knn']               = KNeighborsRegressor(n_neighbors=3)
    models['cart']              = RandomizedSearchCV(DecisionTreeRegressor(),dtree_param,n_jobs=1,n_iter=100,random_state=42)
    models['extra']             = ExtraTreeRegressor(random_state=42)
    models['svmr']              = SVR()

    n_trees = 100
    models['ada']               = AdaBoostRegressor(n_estimators=n_trees,random_state=42)
    models['bag']               = BaggingRegressor(n_estimators=n_trees)
    models['rf']                = RandomForestRegressor(n_estimators=n_trees,random_state=42)
    models['et']                = ExtraTreesRegressor(n_estimators=n_trees,random_state=42)
    models['gbm']               = GradientBoostingRegressor(n_estimators=n_trees,random_state=42)


    return models

def init_ML_models():
    models = dict()
    models['lr']                = LinearRegression()
    models['lasso']             = Lasso()
    models['ridge']             = Ridge()
    models['en']                = ElasticNet()
    models['huber']             = HuberRegressor()
    models['llars']             = LassoLars()
    models['pa']                = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
#    models['knn']               = KNeighborsRegressor(n_neighbors=5)
    models['cart']              = DecisionTreeRegressor()
    models['extra']             = ExtraTreeRegressor()
    models['svmr']              = SVR()

    n_trees = 100
    models['ada']               = AdaBoostRegressor(n_estimators=n_trees)
    models['bag']               = BaggingRegressor(n_estimators=n_trees)
    models['rf']                = RandomForestRegressor(n_estimators=n_trees)
    models['et']                = ExtraTreesRegressor(n_estimators=n_trees)
    models['gbm']               = GradientBoostingRegressor(n_estimators=n_trees)

    return models


def models_arima(order,cluster):
    models = dict()
    if cluster in [2,3,5,6,8,9,20,21,23,24,26,27]:
        #Except for clusters with high sparsity and Non-Seasonal data
        models['ARIMA']              = order
    elif cluster in [1,4,7,10,13,16,19,22,25]:
        #For highly sparse clusters
        models['AR']                 = (order[0], 0, 0)
        models['MA']                 = (0, 0, order[2])
        models['ARMA']               = (order[0],order[2])
    return models


def init_ARIMA_models(order):
    models = dict()
    models['AR']                 = (order[0], 0, 0)
    models['MA']                 = (0, 0, order[2])
    models['ARMA']               = (order[0],order[2])
    models['ARIMA']              = order
    return models


def models_ES(cluster):
    key=[]
#    key=['SES','HWES']
    if cluster in [12,15,18,21,24,27]:
        key=['SES']
    elif cluster in [2,5,8,11,14,17,20,23,26]:
        key=['SES','HWES']
    return key

def init_ES_models():
    models = dict()

    models['SES']               = SimpleExpSmoothing()
    models['HWES']              = ExponentialSmoothing()
    return models


def init_test_shape():
    test_shape_incr = dict()
    test_shape_incr['svmr']     = 0
    test_shape_incr['ada']      = 0
    test_shape_incr['lr']       = 0
    test_shape_incr['lasso']    = 0
    test_shape_incr['ridge']    = 0#1
    test_shape_incr['en']       = 0
    test_shape_incr['huber']    = 0#1
    test_shape_incr['llars']    = 0
    test_shape_incr['pa']       = 0
    test_shape_incr['knn']      = 0
    test_shape_incr['cart']     = 0
    test_shape_incr['extra']    = 3
    test_shape_incr['bag']      = 3#1
    test_shape_incr['rf']       = 3#1
    test_shape_incr['et']       = 3#1
    test_shape_incr['gbm']      = 3#1#

    return test_shape_incr

def test_shape_adder(key):
    if key =='knn':
        test_shape_add=0
    elif key == 'lasso':
        test_shape_add=0
    elif key == 'lr':
        test_shape_add=0
    elif key == 'ridge' or key == 'en' or key == 'huber' or key == 'llars' or key == 'pa':
        test_shape_add=0
    elif key == 'cart' or key == 'extra' or key == 'svmr' or key == 'ada':
        test_shape_add=2
    elif key == 'bag' or key == 'rf' or key == 'et' or key == 'gbm':
         test_shape_add=3
    else:
         test_shape_add=0

    return test_shape_add

#Add predictions and RMSE for all models to json
def add_trained_models_data(sku, expected, rmse, predictions, best_model, interval, best_order):
    sku_data = dict()
    sku_data['sku'] = sku
    sku_data['expected'] = expected
    sku_data['best_model'] = best_model
    sku_data['best_order'] = best_order
    sku_data['interval'] = interval

    predictions_list = []
    for key in rmse.keys():
        model_results = dict()
        model_results['model'] = key
        model_results['rmse'] = rmse[key]
        model_results['predictions'] = predictions[key]
        predictions_list.append(model_results)

    sku_data['predictions'] = predictions_list
    return sku_data


def write_trained_data_to_json(outputs):
    outputs = json.dumps(outputs)
    with open('trained_results.json', 'w', encoding = 'utf-8') as f:
        json.dump(outputs, f, ensure_ascii = False, indent = 4)

def add_forecasted_results(sku, dataset, data, output):
    sku_data = dict()
    sku_data['sku'] = sku
#    sku_data['dataset'] = dataset
#    sku_data['sku_data'] = data

    for key in output:
        sku_data[key] = output[key]

    return sku_data


def output_dataframe(forecast_results, file_name):
    cols = ['sku', 'forecast_period',
            #'error_ml', 'error_ts', 'error_en',
            #'weight_ts', 'weight_ml',
            #'model_ml', 'model_ts',
            'interval', 'last_date', 'forecast_values']#, 'forecast_ml', 'forecast_ts', 'forecast_en'

    output = pd.DataFrame.from_dict(forecast_results)
    output = output.ix[:, cols]

    f = output.forecast_values.apply(pd.Series)
    output = pd.concat([output[:], f[:]], axis=1)
    output.drop('forecast_values', axis = 1, inplace = True)
#

    output.to_csv(file_name)

    return output

def model_ML(dataset = [], tsize = 0, test_shape = 0, model = np.nan, key='', order = (0, 0, 0), train_flag = 0):
    predictions = []
    pred_temp = []
    rmse_val=[]
    scale_flag=0
    if key == 'lr' or key == 'lasso' or key == 'ridge' or key == 'knn' or key == 'svmr':
        scale_flag=1

    if train_flag==1:
        itr=5
        for i in range(3):
            expected=pd.DataFrame(dataset)
            expected=expected.tail(itr).head(3)
            expected=expected.reset_index(drop=True)

            train=dataset[:-itr]

            diff_values = difference(train, order[1])

            if scale_flag==1:
                scaler=scaler_selection(key)
                diff_values=scaler.fit_transform(pd.DataFrame(diff_values).values.reshape(-1,1))

            supervised= timeseries_to_supervised(train, order[0])
            data=supervised.values

            RF_model = fit_model(data, model)
            pred_temp=[]

            for j in range(test_shape):
                X, y = data[:, 0:-1], data[:, -1]
                yhat = forecast_model(RF_model, X)

                forecast=yhat[-1]
                if forecast < 0:
                    forecast = weighted_moving_average(dataset, 1, 3)[0]

                pred_temp.append(forecast)

                train = np.append(train, forecast)

                diff_train = difference(train, order[1])

                if scale_flag==1:
                    scaler=scaler_selection(key)
                    diff_train=scaler.fit_transform(pd.DataFrame(diff_train).values.reshape(-1,1))

                supervised= timeseries_to_supervised(train, order[0])
                data=supervised.values

            pred_temp=pred_temp[1:4]
            plotting(key, pred_temp, expected)

            if i == 2:
                predictions.extend(pred_temp)
            else:
                predictions.append(pred_temp[0])

            rmse_val.append(calculate_rmse(key, expected, pred_temp))
            itr=itr-1

    else:

        dataset_1=copy.deepcopy(dataset)
        diff_values = difference(dataset_1, order[1])

        if scale_flag==1:
            scaler=scaler_selection(key)
            diff_values=scaler.fit_transform(pd.DataFrame(diff_values).values.reshape(-1,1))

        supervised= timeseries_to_supervised(diff_values, order[0])
        data=supervised.values

        RF_model = fit_model(data, model)

        for i in range(test_shape):

            X = data[:, 0:-1]

            yhat = forecast_model(RF_model, X)
#
            forecast=yhat[-1]
            if forecast < 0:
                forecast = weighted_moving_average(data, 1, 3)[0]

            predictions.append(forecast)
            dataset_1 = np.append(dataset_1, forecast)

            diff_values = difference(dataset_1, order[1])

            if scale_flag==1:
                scaler=scaler_selection(key)
                diff_values=scaler.fit_transform(pd.DataFrame(diff_values).values.reshape(-1,1))

            supervised= timeseries_to_supervised(diff_values, order[0])
            data=supervised.values

    predictions = [int(i) for i in predictions]
    return predictions,rmse_val

def model_LinearRegression(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    rmse_val = []


    if train_flag == 1:
        itr = 5
        for i in range(3):
            expected = pd.DataFrame(dataset)
            expected = expected.tail(itr).head(3).reset_index(drop = True)

            train = dataset[:-itr]
            diff_values = difference(dataset, order[1])

            scaler = scaler_selection('lr')
            diff_values = scaler.fit_transform(pd.DataFrame(diff_values).values.reshape(-1, 1))

            supervised = timeseries_to_supervised(diff_values, order[0])
            data = supervised.values

            clf = LinearRegression()
            param = {"fit_intercept": [True, False],
			     "normalize": [False],
			     "copy_X": [True, False]}
            grid = GridSearchCV(clf, param, n_jobs = 1)
            model = fit_model(data, grid)

            for j in range(tsize):
                X = data[:, 0:-1]
                yhat = forecast_model(model, X)


                forecast = yhat[-1]
                if forecast < 0:
                    forecast = weighted_moving_average(dataset, 1, 3)[0]

                predictions.append(forecast)
                train = np.append(train, forecast)
                diff_train = difference(train, order[1])
                diff_train = scaler.fit_transform(pd.DataFrame(diff_train).values.reshape(-1, 1))

                supervised = timeseries_to_supervised(train, order[0])
                data = supervised.values

            predictions = predictions[1:4]
            rmse_val.append(calculate_rmse('GR_LR', expected, predictions))
            itr = itr - 1

    predictions = [int(i) for i in predictions]
    return predictions, rmse_val

def model_SVR_Sigmoid(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = difference(dataset, 1)
        supervised = timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]

        mod = SVR()
        g = list(np.linspace(0.0001,1,25000))
        C = [1]
        param = {"kernel": ["sigmoid"],
			     "gamma": g,
			     "C":C}
        random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
        random_search.fit(X, y)
        clf = SVR(kernel=random_search.best_params_["kernel"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])
        clf.fit(X, y)
        yhat = forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

def model_SVR_RBF(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = difference(dataset, 1)
        supervised = timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]

        mod = SVR()

        g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]

        C = [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10),pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]

        param= {'gamma': g,
			    'kernel': ['rbf'],
			    'C': C}
        grid_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
        grid_search.fit(X, y)
        clf = SVR(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"])
        clf.fit(X, y)
        yhat = forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

def model_SVR_Poly(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = difference(dataset, 1)
        supervised = timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]

        mod = SVR()
        g = list(np.linspace(0.0001,1,1000))
        C = list(np.linspace(0.01,10,25))
        param = {"kernel": ["poly"],
		 	     "degree": range(10,30,1),
			     "gamma": g,
			     "C":C}
        random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
        random_search.fit(X, y)
        clf = SVR(kernel=random_search.best_params_["kernel"],degree=random_search.best_params_["degree"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])

        clf.fit(X, y)
        yhat = forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

def model_DecisionTree(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = difference(dataset, 1)
        supervised = timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]

        dtr = DecisionTreeRegressor()
        param_tree = {"max_depth": [3,None],
				  "min_samples_leaf": sp_randint(1, 11),
				  "criterion": ["mse"],
				  "splitter": ["best","random"],
				  "max_features": ["auto","sqrt",None]}

        gridDT = RandomizedSearchCV(dtr,param_tree,n_jobs=1,n_iter=100)
        gridDT.fit(X, y)
        clf = DecisionTreeRegressor(criterion=gridDT.best_params_["criterion"],splitter=gridDT.best_params_["splitter"],max_features=gridDT.best_params_["max_features"],max_depth=gridDT.best_params_["max_depth"],min_samples_leaf=gridDT.best_params_["min_samples_leaf"])


        clf.fit(X, y)
        yhat = forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

def model_RandomForest(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = difference(dataset, 1)
        supervised = timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]
        rfr = RandomForestRegressor()
        param_forest = {"n_estimators": range(10,1000,100),
        				    "criterion": ["mse"],
        				    "bootstrap": [True, False],
        				    "warm_start": [True, False]
        			}
        gridRF = RandomizedSearchCV(rfr,param_forest,n_jobs=1,n_iter=100)
        gridRF.fit(X, y)
        yhat = forecast_model(gridRF, X)

        inverted = list()
        for i in range(len(yhat)):
            value = inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

def model_Ridge(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = difference(dataset, 1)
        supervised = timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]
        rdg = Ridge()
        para_ridge = {"alpha": list(np.linspace(0.000000001,10000,1000000)),
				  "fit_intercept": [True, False],
				  "normalize": [True, False],
				  "solver": ["auto"]}
        random_rdg = RandomizedSearchCV(rdg, para_ridge, n_jobs=1, n_iter = 100)
        random_rdg.fit(X, y)
        clf = Ridge(alpha=random_rdg.best_params_["alpha"],fit_intercept=random_rdg.best_params_["fit_intercept"],normalize=random_rdg.best_params_["normalize"],solver=random_rdg.best_params_["solver"])

        clf.fit(X, y)
        yhat = forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

def model_Lasso(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = difference(dataset, 1)
        supervised = timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]
        lass = Lasso()
        param_lass = {"alpha": list(np.linspace(0.000000001,100,1000)),
				  "fit_intercept": [True, False],
				  "normalize": [True, False],
				  "selection": ["random"]}
        random_lass = RandomizedSearchCV(lass, param_lass, n_jobs=1, n_iter = 100)
        random_lass.fit(X, y)
        clf = Lasso(alpha=random_lass.best_params_["alpha"], fit_intercept=random_lass.best_params_["fit_intercept"],normalize=random_lass.best_params_["normalize"], selection= random_lass.best_params_["selection"])

        clf.fit(X, y)
        yhat = forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

def model_ElasticNet(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = difference(dataset, 1)
        supervised = timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]
        elas = ElasticNet()
        param = {"alpha": list(np.linspace(0.000000001,100,100000)),
			     "l1_ratio": list(np.linspace(0.000001,100,1000)),
			     "fit_intercept": [True, False],
			     "normalize": [True,False],
			     "precompute": [True, False]}
        random_elas = RandomizedSearchCV(elas, param, n_jobs=1, n_iter= 100)
        random_elas.fit(X, y)
        clf = ElasticNet(alpha = random_elas.best_params_["alpha"], l1_ratio= random_elas.best_params_["l1_ratio"], fit_intercept=random_elas.best_params_["fit_intercept"],
				      normalize=random_elas.best_params_["normalize"], precompute=random_elas.best_params_["precompute"])

        clf.fit(X, y)
        yhat = forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

def model_predict(best_algo, best_order, data, forecast_period,rept=0):

    predictions = []
    ML_models = init_ML_models()
    ARIMA_models = init_ARIMA_models(best_order)

    if best_algo in ML_models.keys():
        if best_algo == 'GR_LR':
            predictions = model_LinearRegression(data.values, forecast_period, best_order)
        elif best_algo == 'SVR_Sigmoid':
            predictions = model_SVR_Sigmoid(data.values, forecast_period, best_order)
        elif best_algo == 'SVR_RBF':
            predictions = model_SVR_RBF(data.values, forecast_period, best_order)
        else:
            test_shape_add=test_shape_adder(best_algo)
            test_shape_fin=forecast_period+test_shape_add
            predictions, rmse = model_ML(dataset = data.values, tsize = forecast_period, test_shape = test_shape_fin, model = ML_models[best_algo], order = best_order)
            if test_shape_add > 0:
                st=test_shape_fin-1
                end=test_shape_add-1
                predictions=predictions[-st:-end]


    elif best_algo in ARIMA_models.keys():
        predictions , rmse = model_ARIMA(best_algo,train = data.values, test_shape = forecast_period, order = ARIMA_models[best_algo],train_flag=0)

    elif best_algo in ['SES', 'HWES']:
        predictions, rmse = model_ES(best_algo, train = data.values, test_shape = forecast_period,train_flag=0)
    elif best_algo in ['naive', 'naive2','naive3','naive6', 'naive12', 'naive12wa','naive6wa']:
        predictions, rmse = model_Naive(best_algo, data.values, forecast_period, best_order,rept,train_flag=0)

    elif best_algo in ['sma', 'wma']:
        predictions, rmse = model_MA(best_algo, data.values, forecast_period, train_flag = 0)
    else:
        predictions = [0]*forecast_period

    return predictions

def find_best_model(outputs):
    best_models = dict()
    for element in outputs:
        sku = element['sku']
        model = element['best_model']
        best_models[sku] = model
    return best_models

def find_intervals(outputs):
    intervals = dict()
    for element in outputs:
        sku = element['sku']
        interval = element['interval']
        intervals[sku] = interval
    return intervals

def plot_forecast(dataset, forecast, color):
    plt.figure(figsize=(30, 7))
    plt.plot(dataset.index, dataset['sales'], 'b-')
    a = len(dataset)
    b = len(forecast)
    index = range(a - 1, a + b - 1)
    plt.plot(index, forecast, color)
    plt.show()

def plot_all_forecasts(dataset, data, forecast, forecast_en, forecast_ml, forecast_ts, sku):
    dataset = dataset.reset_index()
#    data = data.reset_index()
    dataset.columns = ['time', 'sales']
    plt.figure(figsize=(30, 7))
    plt.title(str(sku))
    plt.plot(dataset.index, dataset['sales'], 'y-')
    plt.plot(data, 'b-')
    a = len(dataset)
    b = len(forecast)
    index = range(a - 1, a + b - 1)
    plt.plot(index, forecast, 'k')
    plt.plot(index, forecast_ml, 'r')
    plt.plot(index, forecast_ts, 'c')
    plt.plot(index, forecast_en, 'g')

    plt.show()

def output_forecast(sku, dataset, sku_data, output, forecast_results):
    dataset = dataset.reset_index()
    dataset.columns = ['time', 'sales']

    output['last_date'] = dataset.time.iloc[-1]

    forecast_result = add_forecasted_results(sku, dataset, sku_data, output)
    forecast_results.append(forecast_result)

    with open(r'C:\Users\Alekhya Bhupati\Downloads\SalesForecast\forecast.json', 'w', encoding = 'utf-8') as f:
        json.dump(forecast_results, f, ensure_ascii = False, indent = 4, default = str)

    return forecast_results

def calculate_forecast_accuracy(expected, forecast):
    if math.isnan(expected):
        expected = 0
    else:
        expected = int(expected)
    expected = int(expected)
    forecast = int(forecast)
    print("calculate_forecast_accuracy")
    facc = (1 - (np.abs(expected - forecast)) / (expected+(expected==0))) * 100
    if facc<0:
        facc=0
    mape = (np.abs(expected - forecast) / expected) * 100
    bias = (expected - forecast)

    if np.isnan(facc)== True or np.isfinite(facc)==False:
        facc=0
    if np.isnan(mape)== True or np.isfinite(mape)==False:
        mape=0
    if np.isnan(bias)== True or np.isfinite(bias)==False:
        mape=0
    return float(format(facc,'.3f')), float(format(mape,'.3f')), float(format(bias,'.3f'))

def calculate_validation_facc(expected, predictions):
    validation_facc = []
    for i in range(len(expected)):
        a = int(expected[i])
        b = int(predictions[i])
        value= ((1 - (np.abs(a - b)) /(a+(a==0))) * 100)

        if np.isnan(value)== True or np.isfinite(value)==False:
            value=0
        validation_facc.append(float(format(value,'.3f')))

    validation_facc = [0 if i<0 else i for i in validation_facc]
    print("Validation Accuracy")
    print(validation_facc)
    return validation_facc



def add_outlier_treated_data_to_csv(forecast_results, file_name):
    data = pd.DataFrame()
    for f in forecast_results:
        sku = f['sku']
        data[sku] = f['sku_data'].T.squeeze()

    data = data.T
    data.to_csv(file_name)


def time_series_using_ml(dataset, tsize, order):
#    models = models_ML(cluster)
    models=init_ML_models()
    rmse = dict()
    model_predictions = dict()
#    mape=dict()
#    itr=5
    for key in models.keys():
        test_shape = tsize

        test_shape_incr = init_test_shape()
        if key in test_shape_incr:
            test_shape = test_shape + test_shape_incr[key]
        else:
            test_shape = test_shape + 1

        predictions,rmse_i = model_ML(dataset.values, tsize, test_shape, models[key], key, order, 1)
        predictions=predictions[-tsize:]

        rmse[key]=rmse_i
        model_predictions[key] = predictions

    return model_predictions,rmse

def run_arima_models(data, diff_data, best_order, tsize):

#    models=models_arima(best_order,cluster)
    models = init_ARIMA_models(best_order)

    rmse=dict()
    model_predictions = dict()

    train, test = data[0:-tsize], data[-tsize:]
    test=pd.DataFrame(test)
    test=test.reset_index(drop=True)
    test_shape = len(test)
    for key in models.keys():
        print("KEY!", key)
        if key == 'ARMA' or key == 'AR' or key == 'MA':
            predictions,rmse_i = model_ARMA(key,train, test_shape, models[key], train_flag = 1, test= test)

        else:
            if key == 'ARIMA':
                predictions,rmse_i = model_ARIMA(key,train, test_shape, models[key], train_flag = 1, test= test)

#        mu.plotting(key, predictions, expected)
        rmse[key]=rmse_i

        model_predictions[key] = predictions


    return rmse, model_predictions

def naive_forecast(dataset,freq,p,tsize):
    rmse = dict()
    model_predictions = dict()
    train, test = dataset[0:-tsize], dataset[-tsize:]
    test=pd.DataFrame(test)
    test=test.reset_index(drop=True)
#    expected = test
    test_shape = len(test)
    sam=np.array(dataset)
    repeat=check_repetition(sam, freq , 1, len(sam))

    key='naive'
    predictions,rmse_i = model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions

    key='naive_rept'
    predictions,rmse_i = model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions

    key='naive3'
    predictions,rmse_i = model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions

    key='naive6'
    predictions,rmse_i = model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions

    key='naive12'
    predictions,rmse_i = model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions

    key='naive12wa'
    predictions,rmse_i = model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions



    return rmse,model_predictions

def run_es_models(train, test):
    key=['SES','HWES']
    rmse = dict()
    test_shape = len(test)

    model_predictions = dict()

    if len(key)>0:
        if len(key)==2:
            #SES
            predictions,rmse_i = model_ES(key[0], train, test_shape, train_flag = 1, test = test)
            rmse[key[0]]=rmse_i
            model_predictions[key[0]] = predictions
            #HWES
            predictions,rmse_i = model_ES(key[1], train, test_shape, train_flag = 1, test = test)
            rmse[key[1]]=rmse_i
            model_predictions[key[1]] = predictions

        else:
            predictions,rmse_i = model_ES(key[0], train, test_shape, train_flag = 1, test = test)
            rmse[key[0]]=rmse_i
            model_predictions[key[0]] = predictions

            model_predictions[key[0]] = predictions

    return rmse, model_predictions

def Moving_Average(data,tsize):
    rmse = dict()
    model_predictions = dict()
    key=['sma','wma']
    train, test = data[0:-tsize], data[-tsize:]
    test=pd.DataFrame(test)
    test=test.reset_index(drop=True)
#    expected = test
    test_shape = len(test)
    if len(key)>0:
        if len(key)==2:
            predictions,rmse_i = model_MA(key[0],train, test_shape, train_flag = 1)
            rmse[key[0]]=rmse_i
            model_predictions[key[0]] = predictions

            predictions,rmse_i = model_MA(key[1],train, test_shape, train_flag = 1)
            rmse[key[1]]=rmse_i
            model_predictions[key[1]] = predictions
        else:
            predictions,rmse_i = model_MA(key[0],train, test_shape, train_flag = 1)
            rmse[key[0]]=rmse_i
            model_predictions[key[0]] = predictions

    print("Moving_Average done")

    return rmse,model_predictions

def time_series_models(freq,data, diff_data, tsize, best_order):
    rmse_ARIMA, predictions_ARIMA = run_arima_models(data, diff_data, best_order, tsize)
    rmse_ES, predictions_ES = run_es_models(data, diff_data)
    rmse_naive, predictions_naive = naive_forecast(data,freq,best_order,tsize)
    rmse_ma,predictions_ma = Moving_Average(data,tsize)
    return rmse_ARIMA, rmse_ES, rmse_naive, rmse_ma, predictions_ARIMA, predictions_ES, predictions_naive,predictions_ma

def weight_calculation(data,best_models,best_order):
    itr=5
    weight_ts=0
    weight_ml=0
    for i in range(3):
        print("Running models for ensemble ...",i)
        sample=data[:-itr]
        expected=data.tail(itr).head(3)
        forecast_ml = model_predict(best_models[0], best_order, sample, 3)
        forecast_ts = model_predict(best_models[1], best_order, sample, 3)
        itr-=1
        expected=expected.reset_index(drop=True)
        forecast_ts=pd.DataFrame(forecast_ts)
        rmse_ts=calculate_rmse(best_models[1], expected, forecast_ts)
        rmse_ml=calculate_rmse(best_models[0], expected, forecast_ml)
        weight_ts+=calculate_weight(rmse_ts,rmse_ml)
        weight_ml+=calculate_weight(rmse_ml,rmse_ts)
    weight_ts=weight_ts/3
    weight_ml=weight_ml/3
    return weight_ts,weight_ml

#weights for  ensemble method
def calculate_weight(error1,error2):
    if error1 == 0.0:
        a = 1
        b = 0
    elif error2 == 0.0:
        b =1
        a=0
    else:
        a=1/error1
        b=1/error2
    weight=a/(a+b)
    return weight

def method_ensemble(forecast_ml,forecast_ts, weight_ml, weight_ts,forecast_period):

    forecast=[0]*forecast_period

    for i in range(forecast_period):
        forecast[i]=((weight_ml*forecast_ml[i])+(weight_ts*forecast_ts[i]))/(weight_ml+weight_ts)
        forecast = [0 if i < 0 else int(i) for i in forecast]

    return forecast

'''
Create a differenced series
dataset                 --> data for a sku
interval                --> difference between the two indices
value                   --> difference between values of two indices in datset
diff                    --> list of differenced values
'''
def difference(dataset, interval = 1):
    diff = list()
    if(interval != 0):
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
    else:
        diff = list(dataset)
#        for num in dataset:
#            diff.append(np.asscalar(num))
#        print(diff)

    return pd.Series(diff)

'''
Convert sequence to a supervised learning problem
data                  -->
lag                   -->
dataset               --> data for a sku, converted to a dataframe
columns               -->
'''
def timeseries_to_supervised(dataset, lag =1):
    dataset = pd.DataFrame(dataset)
    y = [dataset.shift(i) for i in range(1, lag + 1)]
    y.append(dataset)
    dataset = pd.concat(y, axis = 1)
    cols = []
    for i in range(lag):
        cols.append('x_' + str(i))
    cols.append('y')
    dataset.columns = cols
#    print(dataset.columns)
    dataset.dropna(axis=0,inplace = True)
    return dataset

def scaler_selection(key):
    if key == 'lr' or key == 'lasso' or key == 'ridge' or key == 'knn':
        scaler = MinMaxScaler(feature_range=(0,1), copy=True)
    elif key == 'svmr':
        scaler = StandardScaler()

    return scaler

'''
Invert differenced values
history              -->
yhat                 -->
interval             -->
'''
def inverse_difference(history, yhat, interval = 1):
    return yhat + history[-interval]

def seasonal_effect(forecast, ft_p, stype, seasonality, j):
    for i in range(ft_p):
        if stype=='additive':
            forecast[i]=forecast[i]+seasonality[j]
        elif stype=='multiplicative':
            forecast[i]=forecast[i]*seasonality[j]
            j+=1
            if j>=len(seasonality)-1:
                j=0
    return forecast

def init_output(forecast_period, raw_data):
    output = {}
    output['forecast_period'] = forecast_period
    output['forecast_values'] = []
    output['interval'] = 'M'
    output['actuals'] = assign_dates(raw_data, 'actuals')
    output['best_models_ml'] = []
    output['best_models_ts'] = []
    output['bias_ml'] = []
    output['bias_ts'] = []
    output['bias_en'] = []
    output['accuracy_ml'] = []
    output['accuracy_ts'] = []
    output['accuracy_en'] = []
    output['validation'] = dict()
    output['validation_facc'] = dict()
    output['facc'] = ''
    output['mape'] = ''
    output['bias'] = ''

    return output

def assign_dates(data, flag, dates = ''):
    if flag == 'validation':
        dates = dates.reset_index()
        dates.columns = ['time', 'sales']
        dates.time = pd.to_datetime(dates.time, format = '%d/%m/%y', infer_datetime_format = True)
        dates.time = dates.time.dt.to_period('M')
        data=[float(format(i,'.3f')) for i in data]
        result = pd.DataFrame({'time': dates.time.astype(str), 'validation': data})
        result.set_index('time', inplace = True)
        result = result.to_dict()
        result = result['validation']

    elif flag == 'val_facc':
        dates = dates.reset_index()
        dates.columns = ['time', 'sales']
        dates.time = pd.to_datetime(dates.time, format = '%d/%m/%y', infer_datetime_format = True)
        dates.time = dates.time.dt.to_period('M')
        result = pd.DataFrame({'time': dates.time.astype(str), 'val_facc': data})
        result.set_index('time', inplace = True)
        result = result.to_dict()
        result = result['val_facc']


    elif flag == 'forecast':
        dates = dates.reset_index()
        dates.columns = ['time', 'forecast']
        last_date = pd.to_datetime(dates.time[0], format = '%d/%m/%y', infer_datetime_format = True)
        print(last_date)
        date_range = pd.date_range(last_date, periods = 6, freq = 'M')
        date_range = date_range.strftime('%Y-%m').tolist()
        date_range.pop(0) #same month as last_date, FUSO
        data=[float(format(i,'.3f')) for i in data]
        result = pd.DataFrame({'time': date_range, 'forecast': data})
        result.set_index('time', inplace = True)
        result = result.to_dict()
        result = result['forecast']


    elif flag == 'actuals':
        data = data.tail(8).reset_index()
        data.columns = ['time', 'sales']
        data['time'] = pd.to_datetime(data['time'], format = '%d/%m/%y', infer_datetime_format = True)
        data['time'] = data['time'].dt.to_period('M').astype(str)
        data['sales'] = [0 if pd.isnull(i) else int(i) for i in data['sales']]
        data['sales']=[float(format(i,'.3f')) for i in data['sales']]
        data.set_index('time', inplace = True)
        result = data.to_dict()
        result = result['sales']

    return result

#%%
os.chdir(r'C:\Users\Alekhya Bhupati\Downloads\SalesForecast')

#%%
#Retrieve Data
forecast_period, datasets = user_input()



#%%
forecast_results = training(datasets, forecast_period)

#%%
output = output_dataframe(forecast_results, "forecast_data.csv")

#%%

def plot_by_years(sku, dataset, forecast, last_date, interval):
    print("SKU ", sku)
    dataset = dataset.reset_index()
    dataset.columns = ['time', 'sales']
    dataset['time'] = pd.to_datetime(dataset['time'], format = '%d/%m/%y', infer_datetime_format = True)
    dataset['time'] = dataset['time'].dt.to_period('M')

    forecast_period = len(forecast)
    date_range = pd.date_range(last_date, periods = forecast_period + 1, freq = interval).tolist()
    date_range.pop(0) #same month as last_date, FUSO
    print(type(date_range))
    date_range = pd.to_datetime(date_range, infer_datetime_format = True)
    print(type(date_range))


    print(forecast)
    data = pd.DataFrame({'time': date_range.strftime('%Y-%m'),
                        'sales' : forecast})
    dataset.append(data)

    print(dataset.info())
    unique_years = dataset.time.year().unique()
    nyears = dataset.time.year().nunqiue()

    if(nyears > 4):
        nyears = 4
        unique_years = unique_years[-4:]

    fig, axes = plt.subplots(nrows = nyears, figsize = (12, 8))
    i = 0
    for row in axes:
        y = dataset[dataset['year'] == unique_years[i]].sales.tolist()
        x = dataset[dataset['year'] == unique_years[i]].month.tolist()
        row.plot(x, y)
        row.scatter(x, y)
        row.set_xlim(1, 12)
#        row.set_xticks(range(12))
        row.set_title("Year " + str(unique_years[i]))
        i += 1
    plt.show()
    print("--------------------")

#%%

