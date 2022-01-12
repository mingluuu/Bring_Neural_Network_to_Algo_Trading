# import open library
import numpy as np
from scipy.stats import mstats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# import data tools
from tools.data_prepare import *
# import indicators library
from tools.generate_indicators import *

def main():
    
    # Initialization
    tickers = ['SPY']
    winsorize = True
    
    # Import data from Yahoo
    data_org = get_data_Yahoo(tickers)
    print(data_org.head(20))
    # Iterate through different window configurations of short window and long window for all indicators
    # short window in range [2:14], with interval of 1
    # long window in range [17:210], with interval of 7
    best_config_return = 0
    for short_window in range(2, 14):
        for long_window in range(7, 210, 7):
            all_data = data_org.copy()
            indicator_proposals = []
            generate_indicators(all_data, indicator_proposals, short_window=short_window, long_window=long_window)
            
            # Create Prediction Variables (Target Value)
            # If the stock went up, denote it by 1,
            # If the stock went down/did not change, denote it by 0.
            all_data['Signals'] = np.where(all_data['return']>=0,1,-1)
            all_data = all_data.dropna().copy()
            
            if winsorize:
                # Winsorizing the Indicators - deal with the outliers in our explanatory variables
                # The idea of winsorizing is to bring extreme outliers to closest value that is not considered as outlier.
                # winsorize the 10% lowest and 10% highest values to the 10th percentile and 90th percentile values respectively.
                target_variables = ['SMA_ratio', 'SMAV_ratio', 'ATR_s', 'ATR_l', 'ATR_ratio', 'ADX_s', 'ADX_l', 'Stochastic_PD_s', 'Stochastic_PD_l', 'Stochastic_ratio', 'RSI_s', 'RSI_l', 'RSI_ratio', 'MACD']
                for variable in target_variables:
                    all_data.loc[:,variable] = mstats.winsorize(all_data.loc[:,variable], limits = [0.1,0.1])
            
            # Iterate through the proposed indicator and compute total strategy return for current window config
            indicators_return_list = []
            for i in range(len(indicator_proposals)):
                
                # Get current indicator set
                indicator_i = indicator_proposals[i]
                
                # Generate train and test sets for machine learning using the current indicator set
                X_train_scaled, X_test_scaled, y_train, y_test, signals_df, X_test = generate_dataset(all_data, indicator_i)
    
                # Initiate the model instance
                decision_tree = DecisionTreeClassifier(random_state=101, max_depth=3)
                # Fit the model using the training data
                model = decision_tree.fit(X_train_scaled,y_train)
                # Use the testing dataset to generate the predictions for the new model
                pred = decision_tree.predict(X_test_scaled)
                # Use a classification report to evaluate the model using the predictions and testing data
#                testing_report = classification_report(y_test, pred)
    
                # Create a predictions DataFrame
                predictions_df = pd.DataFrame(index=X_test.index)
                # Add the SVM model predictions to the DataFrame
                predictions_df['Predicted'] = pred
                # Add the actual returns to the DataFrame
                predictions_df['Actual Returns'] = all_data['return']
                # Add the strategy returns to the DataFrame
                predictions_df['Strategy Returns'] = all_data['return'] * predictions_df['Predicted']
                
                # Compute current indicator's strategy returns
                instance_return = np.product(1 + predictions_df['Strategy Returns'])
                indicators_return_list.append(instance_return)
                
            # Compute the sum of returns for all indicators using current window config
            config_total_return = np.sum(indicators_return_list)
            
            # Compare the current sum of returns with the best recorded
            # If new max found, replace the previous best with current sum of returns
            if config_total_return > best_config_return:
                best_config = [short_window, long_window]
                best_config_return = config_total_return
                print("New best config found!!!")
                print("Current Best Config:")
                print(best_config)
                print("Current Best Config Return:")
                print(best_config_return)
                
    print("Best Config:")
    print(best_config)
    print("Best Config Return:")
    print(best_config_return)
    
    # Use the best config to evaluate the indicators
    short_window, long_window = best_config
    
    all_data = data_org.copy()
    indicator_proposals = []
    generate_indicators(all_data, indicator_proposals, short_window=short_window, long_window=long_window)
    
    # Create Prediction Variables (Target Value)
    # If the stock went up, denote it by 1,
    # If the stock went down/did not change, denote it by 0.
    all_data['Signals'] = np.where(all_data['return']>=0,1,-1)
    all_data = all_data.dropna().copy()
    
    # Review the dataframe and record the indicators
    print(all_data.head(20))
    all_data.to_csv('indicators_records.csv')
    
    if winsorize:
        # Winsorizing the Indicators - deal with the outliers in our explanatory variables
        # The idea of winsorizing is to bring extreme outliers to closest value that is not considered as outlier.
        # winsorize the 10% lowest and 10% highest values to the 10th percentile and 90th percentile values respectively.
        target_variables = ['SMA_ratio', 'SMAV_ratio', 'ATR_s', 'ATR_l', 'ATR_ratio', 'ADX_s', 'ADX_l', 'Stochastic_PD_s', 'Stochastic_PD_l', 'Stochastic_ratio', 'RSI_s', 'RSI_l', 'RSI_ratio', 'MACD']
        for variable in target_variables:
            all_data.loc[:,variable] = mstats.winsorize(all_data.loc[:,variable], limits = [0.1,0.1])
            
    for i in range(len(indicator_proposals)):
                
        # Get current indicator set
        indicator_i = indicator_proposals[i]
        
        # Generate train and test sets for machine learning using the current indicator set
        X_train_scaled, X_test_scaled, y_train, y_test, signals_df, X_test = generate_dataset(all_data, indicator_i)
        
        # Initiate the model instance
        decision_tree = DecisionTreeClassifier(random_state=101, max_depth=3)
        # Fit the model using the training data
        model = decision_tree.fit(X_train_scaled,y_train)
        # Use the testing dataset to generate the predictions for the new model
        pred = decision_tree.predict(X_test_scaled)
        # Use a classification report to evaluate the model using the predictions and testing data
#       testing_report = classification_report(y_test, pred)
        
        # Create a predictions DataFrame
        predictions_df = pd.DataFrame(index=X_test.index)
        # Add the SVM model predictions to the DataFrame
        predictions_df['Predicted'] = pred
        # Add the actual returns to the DataFrame
        predictions_df['Actual Returns'] = all_data['return']
        # Add the strategy returns to the DataFrame
        predictions_df['Strategy Returns'] = all_data['return'] * predictions_df['Predicted']
        
        # Compute current indicator's strategy returns
        instance_return = np.product(1 + predictions_df['Strategy Returns'])
                
        # Plot the actual returns versus the strategy returns
        plot_name = 'DTree_'+indicator_i[0]+'_short'+str(short_window)+'_long'+str(long_window)+'_'+str(round(instance_return,2))+'_Plot'
        DTree_SMA_best_plot = (1 + predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod().plot(title=plot_name).get_figure()
        DTree_SMA_best_plot.savefig('./plots/'+plot_name+'.png')
    
if __name__ == '__main__':
    main()
