# *Project 2 Team 6: Bring Neural Network to Algo Trading Report* 
---
**Team members: 
Ziwen Chen,
Andy He, 
Shasha Li, 
Minglu Li**

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

![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/Data%20Prep.png)
![alt text](https://github.com/Z1WenChen/Project_2/blob/main/Files/Data%20Prep.png)


**Solution:**
*1) Only select 4 indicators for momentum and non-momentum set respectively*
*2) Encode the indicators and create categorical variables to make indicators more “significant” to the signal and train scaled Xs*
*3) Activation function change to “elu” for hidden layers to produce negative output*
*4) Activation function change to “linear” for the output layer*
*5) Loss function change to “mse”*
*6) Metrics change to “mse”*




------------------------------------------------------------------------------------------------------------


