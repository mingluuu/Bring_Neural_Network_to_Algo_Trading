#importing variables
import numpy as np

def wilder_smooth(dataslice_df, periods):
    '''
    Function to perform wilder smoothing, a weighted moving average
    Inputs:
        dataslice_df: a dataframe with a slice of data.
        periods: an int specify the smoothing window (days).
    Return:
        Wilder: an numpy array of the smoothed dataslice.
    '''
    # Check if nans present in beginning
    start = np.where(~np.isnan(dataslice_df))[0][0]
    
    Wilder = np.array([np.nan]*len(dataslice_df))
    #Simple Moving Average
    Wilder[start+periods-1] = dataslice_df[start:(start+periods)].mean()
    #Wilder Smoothing
    for i in range(start+periods,len(dataslice_df)):
        Wilder[i] = (Wilder[i-1]*(periods-1) + dataslice_df[i])/periods 
    
    return Wilder

def add_indicator_SMA(data_df, short_window=5, long_window=15):
    '''
    Function to compute and add Simple Moving Average (SMA) as indicators into the input dataframe.
    Inputs:
        data_df: a dataframe with stock information.
        short_window: an int specify the SHORT term window, default set to 5 days.
        long_window: an int specify the LONG term window, default set to 15 days.
    '''
    data_df['SMA_s'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=short_window).mean())
    data_df['SMA_l'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=long_window).mean())
    data_df['SMA_ratio'] = data_df['SMA_s'] / data_df['SMA_l']
    
def add_indicator_SMAV(data_df, short_window=5, long_window=15):
    '''
    Function to compute and add Simple Moving Average Volume (SMAV) as indicators into the input dataframe.
    Inputs:
        data_df: a dataframe with stock information.
        short_window: an int specify the SHORT term window, default set to 5 days.
        long_window: an int specify the LONG term window, default set to 15 days.
    '''
    data_df['SMAV_s'] = data_df.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window=short_window).mean())
    data_df['SMAV_l'] = data_df.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window=long_window).mean())
    data_df['SMAV_ratio'] = data_df['SMAV_s'] / data_df['SMAV_l']
    
def add_indicator_ATR(data_df, short_window=5, long_window=15):
    '''
    Function to compute and add Average True Range (ATR) as indicators into the input dataframe.
    Inputs:
        data_df: a dataframe with stock information.
        short_window: an int specify the SHORT term window, default set to 5 days.
        long_window: an int specify the LONG term window, default set to 15 days.
    '''
    data_df['prev_close'] = data_df.groupby('symbol')['Close'].shift(1)
    data_df['TR'] = np.maximum((data_df['High'] - data_df['Low']), 
                         np.maximum(abs(data_df['High'] - data_df['prev_close']), 
                         abs(data_df['prev_close'] - data_df['Low'])))
    for i in data_df['symbol'].unique():
        TR_data = data_df[data_df.symbol == i].copy()
        data_df.loc[data_df.symbol==i,'ATR_s'] = wilder_smooth(TR_data['TR'], short_window)
        data_df.loc[data_df.symbol==i,'ATR_l'] = wilder_smooth(TR_data['TR'], long_window)

    data_df['ATR_ratio'] = data_df['ATR_s'] / data_df['ATR_l']
    
def add_indicator_ADX(data_df, short_window=5, long_window=15):
    '''
    Function to compute and add Average Directional Index (ADX) as indicators into the input dataframe.
    Inputs:
        data_df: a dataframe with stock information.
        short_window: an int specify the SHORT term window, default set to 5 days.
        long_window: an int specify the LONG term window, default set to 15 days.
    '''
    data_df['prev_high'] = data_df.groupby('symbol')['High'].shift(1)
    data_df['prev_low'] = data_df.groupby('symbol')['Low'].shift(1)

    data_df['+DM'] = np.where(~np.isnan(data_df.prev_high),
                        np.where((data_df['High'] > data_df['prev_high']) & 
                        (((data_df['High'] - data_df['prev_high']) > (data_df['prev_low'] - data_df['Low']))), 
                        data_df['High'] - data_df['prev_high'], 0), np.nan)

    data_df['-DM'] = np.where(~np.isnan(data_df.prev_low),
                        np.where((data_df['prev_low'] > data_df['Low']) & 
                        (((data_df['prev_low'] - data_df['Low']) > (data_df['High'] - data_df['prev_high']))), 
                        data_df['prev_low'] - data_df['Low'], 0), np.nan)

    for i in data_df['symbol'].unique():
        ADX_data = data_df[data_df.symbol == i].copy()
        data_df.loc[data_df.symbol==i,'+DM_s'] = wilder_smooth(ADX_data['+DM'], short_window)
        data_df.loc[data_df.symbol==i,'-DM_s'] = wilder_smooth(ADX_data['-DM'], short_window)
        data_df.loc[data_df.symbol==i,'+DM_l'] = wilder_smooth(ADX_data['+DM'], long_window)
        data_df.loc[data_df.symbol==i,'-DM_l'] = wilder_smooth(ADX_data['-DM'], long_window)

    data_df['+DI_s'] = (data_df['+DM_s']/data_df['ATR_s'])*100
    data_df['-DI_s'] = (data_df['-DM_s']/data_df['ATR_s'])*100
    data_df['+DI_l'] = (data_df['+DM_l']/data_df['ATR_l'])*100
    data_df['-DI_l'] = (data_df['-DM_l']/data_df['ATR_l'])*100

    data_df['DX_s'] = (np.round(abs(data_df['+DI_s'] - data_df['-DI_s'])/(data_df['+DI_s'] + data_df['-DI_s']) * 100))

    data_df['DX_l'] = (np.round(abs(data_df['+DI_l'] - data_df['-DI_l'])/(data_df['+DI_l'] + data_df['-DI_l']) * 100))

    for i in data_df['symbol'].unique():
        ADX_data = data_df[data_df.symbol == i].copy()
        data_df.loc[data_df.symbol==i,'ADX_s'] = wilder_smooth(ADX_data['DX_s'], short_window)
        data_df.loc[data_df.symbol==i,'ADX_l'] = wilder_smooth(ADX_data['DX_l'], long_window)
        
def add_indicator_STO(data_df, short_window=5, long_window=15):
    '''
    Function to compute and add Stochastic Oscillators (STO) as indicators into the input dataframe.
    Inputs:
        data_df: a dataframe with stock information.
        short_window: an int specify the SHORT term window, default set to 5 days.
        long_window: an int specify the LONG term window, default set to 15 days.
    '''
    data_df['Lowest_s'] = data_df.groupby('symbol')['Low'].transform(lambda x: x.rolling(window=short_window).min())
    data_df['High_s'] = data_df.groupby('symbol')['High'].transform(lambda x: x.rolling(window=short_window).max())
    data_df['Lowest_l'] = data_df.groupby('symbol')['Low'].transform(lambda x: x.rolling(window=long_window).min())
    data_df['High_l'] = data_df.groupby('symbol')['High'].transform(lambda x: x.rolling(window=long_window).max())

    data_df['Stochastic_s'] = ((data_df['Close'] - data_df['Lowest_s'])/(data_df['High_s'] - data_df['Lowest_s']))*100
    data_df['Stochastic_l'] = ((data_df['Close'] - data_df['Lowest_l'])/(data_df['High_l'] - data_df['Lowest_l']))*100

    data_df['Stochastic_PD_s'] = data_df['Stochastic_s'].rolling(window=short_window).mean()
    data_df['Stochastic_PD_l'] = data_df['Stochastic_l'].rolling(window=long_window).mean()

    data_df['Stochastic_ratio'] = data_df['Stochastic_PD_s']/data_df['Stochastic_PD_l']
    
def add_indicator_RSI(data_df, short_window=5, long_window=15):
    '''
    Function to compute and add Relative Strength Index (RSI) as indicators into the input dataframe.
    Inputs:
        data_df: a dataframe with stock information.
        short_window: an int specify the SHORT term window, default set to 5 days.
        long_window: an int specify the LONG term window, default set to 15 days.
    '''
    data_df['Diff'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.diff())
    data_df['Up'] = data_df['Diff']
    data_df.loc[(data_df['Up']<0), 'Up'] = 0

    data_df['Down'] = data_df['Diff']
    data_df.loc[(data_df['Down']>0), 'Down'] = 0 
    data_df['Down'] = abs(data_df['Down'])

    data_df['avg_up_s'] = data_df.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=short_window).mean())
    data_df['avg_down_s'] = data_df.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=short_window).mean())

    data_df['avg_up_l'] = data_df.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=long_window).mean())
    data_df['avg_down_l'] = data_df.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=long_window).mean())

    data_df['RS_s'] = data_df['avg_up_s'] / data_df['avg_down_s']
    data_df['RS_l'] = data_df['avg_up_l'] / data_df['avg_down_l']

    data_df['RSI_s'] = 100 - (100/(1+data_df['RS_s']))
    data_df['RSI_l'] = 100 - (100/(1+data_df['RS_l']))

    data_df['RSI_ratio'] = data_df['RSI_s']/data_df['RSI_l']
    
def add_indicator_MACD(data_df, short_window=5, long_window=15):
    '''
    Function to compute and add Moving Average Convergence Divergence (MACD) as indicators into the input dataframe.
    Inputs:
        data_df: a dataframe with stock information.
        short_window: an int specify the SHORT term window, default set to 5 days.
        long_window: an int specify the LONG term window, default set to 15 days.
    '''
    data_df['Ewm_s'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=short_window, adjust=False).mean())
    data_df['Ewm_l'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=long_window, adjust=False).mean())
    data_df['MACD'] = data_df['Ewm_l'] - data_df['Ewm_s']
    
def add_indicator_BB(data_df, short_window=5, long_window=15):
    '''
    Function to compute and add Bollinger Bands (BB) as indicators into the input dataframe.
    Inputs:
        data_df: a dataframe with stock information.
        short_window: an int specify the SHORT term window, default set to 5 days.
        long_window: an int specify the LONG term window, default set to 15 days.
    '''
    data_df['MA_s'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=short_window).mean())
    data_df['SD_s'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=short_window).std())
    
    data_df['MA_l'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=long_window).mean())
    data_df['SD_l'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=long_window).std())
    
    data_df['lowerband_s'] = data_df['MA_s'] - 2*data_df['SD_s']
    data_df['upperband_s'] = data_df['MA_s'] + 2*data_df['SD_s']
    
    data_df['lowerband_l'] = data_df['MA_l'] - 2*data_df['SD_l']
    data_df['upperband_l'] = data_df['MA_l'] + 2*data_df['SD_l']
    
def add_indicator_RC(data_df, short_window=5, long_window=15):
    '''
    Function to compute and add Rate of Change (RC) as indicators into the input dataframe.
    Inputs:
        data_df: a dataframe with stock information.
        short_window: an int specify the SHORT term window, default set to 5 days.
        long_window: an int specify the LONG term window, default set to 15 days.
    '''
    data_df['RC_s'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.pct_change(periods=short_window))
    data_df['RC_l'] = data_df.groupby('symbol')['Close'].transform(lambda x: x.pct_change(periods=long_window))
