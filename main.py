# import open library
from scipy.stats import mstats

# import data tools
from tools.data_prepare import get_data_Yahoo
# import indicators library
from tools.generate_indicators import *

def main():
    
    # Initialize data and define windows
    tickers = ['SPY']
    short_window = 5
    long_window = 15
    
    # Import data from Yahoo
    all_data = get_data_Yahoo(tickers)
    
    # Generate Inicators
    # generate SMA_s, SMA_l, SMA_ratio
    add_indicator_SMA(all_data, short_window=short_window, long_window=long_window)
    # generate SMAV_s, SMAV_l, SMAV_ratio
    add_indicator_SMAV(all_data, short_window=short_window, long_window=long_window)
    # generate ATR_s, ATR_l, ATR_ratio
    add_indicator_ATR(all_data, short_window=short_window, long_window=long_window)
    # generate ADX_s, ADX_l
    add_indicator_ADX(all_data, short_window=short_window, long_window=long_window)
    # generate Stochastic_PD_s, Stochastic_PD_l, Stochastic_ratio
    add_indicator_STO(all_data, short_window=short_window, long_window=long_window)
    # generate RSI_s, RSI_l, RSI_ratio
    add_indicator_RSI(all_data, short_window=short_window, long_window=long_window)
    # generate MACD
    add_indicator_MACD(all_data, short_window=short_window, long_window=long_window)
    # generate lowerband_s, upperband_s, lowerband_l, upperband_l
    add_indicator_BB(all_data, short_window=short_window, long_window=long_window)
    # generate RC_s, RC_l
    add_indicator_RC(all_data, short_window=short_window, long_window=long_window)
    
    # Create Prediction Variables (Target Value)
    # Technical Indicators generally work well in short interval predictions, 
    # and since our indicators have based on 5-day and 15-day periods, 
    # a 7 (trading) days prediction interval is used.
    # If the stock went up in 7 days, denote it by 1,
    # If the stock went down/did not change, denote it by 0.
    # In code, we shift the price by 6 days.
    all_data['Close_Shifted'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.shift(-6))
    all_data['Target'] = ((all_data['Close_Shifted'] - all_data['Open'])/(all_data['Open']) * 100).shift(-1)
    all_data['Signals'] = np.where(all_data['Target']>0,1,0)
    all_data = all_data.dropna().copy()
    
    # Winsorizing the Indicators - deal with the outliers in our explanatory variables
    # The idea of winsorizing is to bring extreme outliers to closest value that is not considered as outlier.
    # winsorize the 10% lowest and 10% highest values to the 10th percentile and 90th percentile values respectively.
    target_variables = ['SMA_ratio', 'SMAV_ratio', 'ATR_s', 'ATR_l', 'ATR_ratio', 'ADX_s', 'ADX_l', 'Stochastic_PD_s', 'Stochastic_PD_l', 'Stochastic_ratio', 'RSI_s', 'RSI_l', 'RSI_ratio', 'MACD']
    for variable in target_variables:
        all_data.loc[:,variable] = mstats.winsorize(all_data.loc[:,variable], limits = [0.1,0.1])
    
    # Review the dataframe
    print(all_data.head(20))
    
if __name__ == '__main__':
    main()
