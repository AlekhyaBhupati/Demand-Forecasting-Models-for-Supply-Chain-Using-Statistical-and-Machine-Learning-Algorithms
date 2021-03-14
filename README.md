# Demand-Forecasting-Models-for-Supply-Chain-Using-Statistical-and-Machine-Learning-Algorithms

Demand Forecasting is one of the crucial elements of any organisation’s Supply Chain Management (SCM) which helps demand planners to predict the future forecasts. In this analysis the dataset used is of a USA lighting manufacturing company. These datasets are provided by Analytic Labs Research group of India. In this the most important is the pattern of the sales which is needed to check how actually the sales comes out in the future taking into the sense of the validation and history of the sales over three years (Nov 2013 to August 2017). Here the SKUs (Stock Keeping Units) that are used have different patterns, like NPI (New product introduction), which is a new item being introduced into the market which has a less amount of sales history. Their quantity is high as a new product and due to immense marketing strategies involved in the beginning, the sales is high. There are also seasonal products, growing trends products, declining trends products, sparse data product etc. which adds to the variety of the sales pattern of the data. Also, we have used various Time Series (TS) and Machine Learning (ML) algorithms in our model which draws best algorithm with less error from each technique respectively to forecast the sales, we observed that the proposed decision integration strategy of ensemble of algorithm improved the forecast accuracy for Seasonal products, damping non-sparse products and normal sales products, but ensemble algorithm doesn’t work for Sparsely data products as there is very less value of sales data which can’t be tracked off for predictive analysis.

**Introduction**

Demand Forecasting can help business to increase their sales or productivity. There is no end for needs of humans and always we want everything to be done in quick and easier way. So, there is happening lot of innovations in every field of business [19]. With help of this technology there are lot of business or companies are started in small scale, large scale, online business etc… to fulfil the needs of the people leading to a huge competition in the market and getting lots and lots of data. Here arises the need for processing the data effectively with good forecasting, which will help the businesses to stand in the market with good profits.

The purpose of this document is to build a real time application for forecasting the demand analysis for all types of products or SKUs (Stock Keeping Unit) with more accurate forecast values by using different times series and regression, Machine Learning and ensemble algorithms.

**Research Question**

   _“To what extent the proposed decision integration strategy able to forecast the time periods of different sales patterns of products/SKUs (sparsely products, new introduction products, seasonal products and seasonal damping products) data using both time series(AR, ARMA, ARIMA, SMA, SES, HWES) and machine learning algorithms (lr, lasso, en, ridge, llars, pa, cart, knn) by ensemble approach (ada, bag, rf, et, gbm) without effecting the accuracy and cyclicity of the data?”_

There is a significant need for forecasting the sales data with respective to the pattern of sales. Suppose there is a festival season like Diwali, Christmas and Ramzan there will be an increase in the sales due to different seasonal offers, producing a variety of sales patterns, then we must forecast the result accordingly. In the same way for growing trend products, sparsely products there will be a different pattern of sales. To forecast sales of pattern, different algorithms or techniques are needed. But to implement different algorithms for every different pattern is time consuming. So, here we developed a single model which can pass all these different patterns of SKUs data through the system and gives the best forecast accuracy.

**Research Objectives and Contributions Demand Forecasting Models for Supply Chain**

_**Objective 1**_ - Data Acquisition and Data Preparation – Acquiring the time frame for the given SKUs. Doing the ABC Classification to identify the key SKUs. And performing Auto correlation and Partial Correlation on data.

_**Objective 2**_ - Implementation of timeseries algorithms like Auto Regressive, Simple Moving Average, Auto Regressive Moving average (ARMA), Auto Regressive Integrating Moving average, Simple Exponential Smoothing and Holts-Winter Exponential Smoothing model.
RMSE, MAPE, Forecast accuracy

_**Objective 3**_ - Implementation Machine learning algorithms like Linear Regression, Lasso, Ridge, Elastic Net, Huber, Lasso Lars and Passive Aggressive Models, K-Neighbors, Decision Tree, Extra Tree and SVMR.
RMSE, MAPE, Forecast accuracy 

_**Objective 4**_ - Implementation of ensembles using AdaBoost, Bagging, Random Forest, Extra Trees and Gradient Boosting regressor models.
RMSE, MAPE, Forecast accuracy

_**Objective 5**_ - Implementation, Evaluation and results of sales forecasting for SKUs.
RMSE, MAPE, Forecast accuracy

_**Objective 6**_ - Comparison of forecast results for all the SKUs.
