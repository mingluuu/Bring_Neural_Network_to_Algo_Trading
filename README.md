# *Bring Neural Network to Algo Trading Report* 

------------------------------------------------------------------------------------------------------------


**Purpose of the Project**

*1) Evaluate two groups of technical indicators, Momentum vs Non-momentum, in Neural Network model through comparing model results of predicting SPY’s Up / Flat / Down*

*2) Implement those indicators into Algorithm Trading with ML model, automatically perform Buy / Hold / Sell trades  and backtesting them.*

*3) Our project is a prototype for further application and research, such as apply into different sectors and can be replaced by other indicators*



------------------------------------------------------------------------------------------------------------

**Two Parts of the Project**

*1) Apply sequential model to study momentum and Non-momentum technical indicators with the signal, and compare the results from these two models to identify which set outperforms in terms of loss and accuracy*


![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/Deep%20Learning.png)

*2) Using SVC classifier model to backtest the algo trading returns from momentum and Non-momentum indicators to demonstrate which set is more “accurate” and “profitable”.*

![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/Backtesting.png)





*3) Selection of Momentum and Non-Momentum sets:*

*Momentum set:*

*1. Relative Strength Index “RSI”*

*2. Commodity Channel Index “CCI”*

*3. Rate-of-change “ROC”*

*4. Stochastic Oscillator %K “STOCH”*



*Non-Momentum set:*

*1) Volume Weighted Average Price “VWAP”*

*2) Exponential Moving Average “EMA” (9, 70)*

*3) Bollinger Bands “BBANDS”*

*4) Elder’s Force Index “EFI”*



------------------------------------------------------------------------------------------------------------

**Data Preparation**

*Daily ohlc and volume data of SPY from 2015-01-01 to today through yfinance API*

*Using the daily return of SPY to create Buy, Sell, Hold Signal, assuming 1% commission*

*Notice: This is a multi-class classification problem rather than a binary cross-entropy as suggested by Prof. Vini.*

*Notice: For multi-class classification:
activation = “softmax”, 
loss function = “categorical_crossentropy”*

![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/Data%20Prep.png)

------------------------------------------------------------------------------------------------------------

**Part 1**

*Initially, we select and create 15 technical indicators through FINTA library, and use neural network to study them with the signal, hoping to find significant indicators through iterating the model by trimming the indicators.*


*But we met recurrent neural network input problems caused by*

*1)  indicators are littlely changed in numbers*

*2)  indicators are too many and too complicated*

![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/Neural%20Network%20Deadend.png)
![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/Neural%20Network%20Deadend%20-2.png)


**Solution:**

*1) Only select 4 indicators for momentum and non-momentum set respectively*

*2) Encode the indicators and create categorical variables to make indicators more “significant” to the signal and train scaled Xs*

*3) Activation function change to “elu” for hidden layers to produce negative output*

*4) Activation function change to “linear” for the output layer*

*5) Loss function change to “mse”*

*6) Metrics change to “mse”*

![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/nnmodel.png)


**Results:**

*Please review the codes "Trend_Indicators.ipynb" and "momentum_indicators.ipynb" for results*

**Momentum X_train:**
![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/momxtrain.png)

**Non-Momentum X_train:**
![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/nonmomxtrain.png)

**Momentum X_test:**
![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/momxtest.png)

**Non-Momentum X_test:**
![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/nonmomxtest.png)

*Finding: The Momentum indicators, with both lower Loss and MSE, outperformed the Non-momentum indicators*


------------------------------------------------------------------------------------------------------------

**Part 2**

*Then, we apply momentum and non-momentum set into SVC classifier model and backtest the algo return vs actual return*

*1) 3 Months Training Period*

*2) Standard Scaler*

*3) RandomOverSampler to resample as ramdom_state=1*



**Results:**

*Please review the codes "Trend_Indicators_SVC_test.ipynb" and "SVC_Momentum_Test.ipynb" for results*

**Momentum y_resampled:**
![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/momyresample.png)

**Non-Momentum y_resampled:**
![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/nonmomyresample.png)

**Momentum y_test:**
![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/momytest.png)

**Non-Momentum y_test:**
![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/nonmomytest.png)

*Finding 1: Through running SVC model, non-momentum and momentum models yield close results; but overall, non-momentum outperformed momentum set*

*Finding 2: Through running SVC model, non-momentum outperformed momentum set in algorithm return; but both sets underperformed than the actual return*

**Momentum:**

![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/momreturn.png)

**Non-Momentum:**

![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/nonmomreturn.png)


------------------------------------------------------------------------------------------------------------

**Summary**

![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/summary.png)

**1) Momentum indicators outperform non-momentum in neural network with lower loss and mse, so momentum indicators might have more “explaining power” to trading signals**

**2) Non-momentum indicators outperform momentum in machine learning backtesting with higher f1-score and algorithm returns, so non-momentum indicators might have higher “applicable score” to trading signals**

**3) The algorithm trading returns from both sets are underperformed than the actual returns, so generally speaking, applying technical indicators into algorithm trading is still ambiguous**

------------------------------------------------------------------------------------------------------------

**Next Steps**

*1) Test more indicators in Algo trading*

*2) Run ML-backtesting to each technical indicator*

*3) Optimize our model by testing more Classifiers (eg. Logistic_Regression)*

*4) Apply the method into different sectors and stocks/ETFs*

------------------------------------------------------------------------------------------------------------

**Comments**

*Comment 1: Use Long-Short Term Memory from Keras (LSTM) to make the model move from local minimum*

*Comment 2: 1% Commission might be the reason causing the algo returns underperforming, but it mimics the real world.*

------------------------------------------------------------------------------------------------------------

**Contributers**
Minglu (Amber) Li,
Ziwen Chen,
Shasha Li,
Andy He
