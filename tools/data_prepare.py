import pandas as pd
import datetime as dt
import pandas_datareader as pdr

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
