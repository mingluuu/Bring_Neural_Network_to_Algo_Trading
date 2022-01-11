import pandas as pd
import datetime as dt
import pandas_datareader as pdr
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler

def get_data_Yahoo(tickers, start_year=2017, start_month=1, start_day=1):
    '''
    Function to get the up-to-date stock information from Yahoo Finance API.
    Inputs:
        tickers: an array to specify stock names.
        start_year: an int to specify start year, default set to 2017.
        start_month: an int to specify start month, default set to 1 (Jan).
        start_day: an int to specify start day, default set to 1.
    Return:
        all_data: a dataframe of the requested stock information, with the daily percentage returns.
    '''
    
    # Initialization
    all_data = pd.DataFrame()
    test_data = pd.DataFrame()
    no_data = []

    # Iterate through tickers array and get the data
    for i in tickers:
        try:
            test_data = pdr.get_data_yahoo(i, start=dt.datetime(start_year,start_month,start_day), end=dt.date.today())
            test_data['symbol'] = i
            all_data = all_data.append(test_data)
        except:
            no_data.append(i)

    #Creating Return column
    all_data['return'] = all_data.groupby('symbol')['Close'].pct_change()

    return all_data
    
def generate_dataset(all_data, indicator_list):
    # Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
    X = all_data[indicator_list].shift().dropna()
    # Create the target set selecting the Signal column and assiging it to y
    y = all_data['Signals']
    training_begin = X.index.min()
    # Generate the X_train and y_train DataFrames
    training_end = X.index.min() + DateOffset(months=3)
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]
        # Generate the X_test and y_test DataFrames
    X_test = X.loc[training_end+DateOffset(hours=1):]
    y_test = y.loc[training_end+DateOffset(hours=1):]
    # Create a StandardScaler instance
    scaler = StandardScaler()
    # Apply the scaler model to fit the X-train data
    X_scaler = scaler.fit(X_train)
    # Transform the X_train and X_test DataFrames using the X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, all_data, X_test
