import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import os
import sys
from pandas_market_calendars import get_calendar
from datetime import datetime, timedelta

#_______________________________________Load___________________________________________________
def predict(cd,stock,stockLocation, size_graph):

    # Load the trained model
    modelHL = load_model(cd + "\\Model\\" + stockLocation +"HLTrainedLSTM.h5")
    modelClose = load_model(cd + "\\Model\\" + stockLocation +"CloseTrainedLSTM.h5")


    # Load the test set features and labels from numpy arrays
    X_test_close = np.load(cd + "\\npy\\" + stockLocation +"X_test_close.npy")
    y_test_close = np.load(cd + "\\npy\\" + stockLocation +"y_test_close.npy")
    dates = np.load(cd + "\\npy\\" + stockLocation +"dates_close.npy")

    X_test_HL = np.load(cd + "\\npy\\" + stockLocation +"X_test_HL.npy")
    y_test_HL = np.load(cd + "\\npy\\" + stockLocation +"y_test_HL.npy")


    #_______________________________________Predicting Test for accuracy___________________________________________________

    # Load the scaler object for y,x for predicting
    scaler_y_close = joblib.load(cd + "\\Scaler\\" +stockLocation +"scaler_y_close.pkl")
    scaler_X_close = joblib.load(cd + "\\Scaler\\" +stockLocation +"scaler_X_close.pkl")

    # Evaluate
    accuracy, mae,mse = modelClose.evaluate(X_test_close,y_test_close)

    # Prediction
    predictions_close = modelClose.predict(X_test_close)

    # Inverse scaler for prediction
    prediction_close = scaler_y_close.inverse_transform(predictions_close)

    # Scalers the actual back
    Actual_close = scaler_y_close.inverse_transform(y_test_close)

    # Used for later to only compare dates in test set
    dates_comparsion = []

    # Redirecting output to a file
    sys.stdout = open(cd + "\\PredictionTestSet" + stockLocation + 'CloseTestPredictions.txt', 'w')

    # Adds the correct dates from the dates npy file so we have correct dates
    for i in range(y_test_close.size):
        # Prints date 
        print("\n", dates[i])
        dates_comparsion.append(dates[i])
        print("Prediction: ", prediction_close[i][0])  
        print("Actual: ", Actual_close[i][0]) 


    # Test accuracy
    print(accuracy)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", np.sqrt(mse))

    # Close the file
    sys.stdout.close()

    # Restore stdout to the default value
    sys.stdout = sys.__stdout__

    # Plot values Vs Actual
    plt.figure(figsize=(15, 8))
    plt.plot(dates_comparsion, prediction_close, c="r", label="Predictions", marker="+")
    plt.plot(dates_comparsion, Actual_close, c="g", label="Actual", marker="^")
    plt.title(stock)
    plt.legend()
    plt.tight_layout()
    ifexists = cd + "\\Imagies" + stockLocation + 'Close Actual VS Predictions.png'

    # Remove the existing file if it exists
    if os.path.exists(ifexists):
        os.remove(ifexists)

    # Save the plot with the new filename
    plt.savefig(cd + "\\Imagies" + stockLocation + 'Close Actual VS Predictions.png')

    #________________________________________HighLow_______________________________________________________

    # Load the scaler object for y,x for predicting
    scaler_y_HL = joblib.load(cd + "\\Scaler\\" +stockLocation +"scaler_y_HL.pkl")
    scaler_X_HL = joblib.load(cd + "\\Scaler\\" +stockLocation +"scaler_X_HL.pkl")

    # Evaluate
    accuracy, mae,mse = modelHL.evaluate(X_test_HL,y_test_HL)

    # Prediction
    predictions_HL = modelHL.predict(X_test_HL)

    # Inverse scaler for prediction
    predictions_HL = scaler_y_HL.inverse_transform(predictions_HL)

    # Scalers the actual back
    Actual_HL = scaler_y_HL.inverse_transform(y_test_HL)

    # Used for later to only compare dates in test set
    dates_comparsion = []

    # Redirecting output to a file
    sys.stdout = open(cd + "\\PredictionTestSet" + stockLocation + 'H CloseTestPredictions.txt', 'w')

    # Adds the correct dates from the dates npy file so we have correct dates
    for i in range(y_test_close.size):
        # Prints date 
        print("\n", dates[i])
        dates_comparsion.append(dates[i])
        print("Prediction: ", predictions_HL[i])  
        print("Actual: ", Actual_HL[i]) 


    # Test accuracy
    print(accuracy)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", np.sqrt(mse))

    # Close the file
    sys.stdout.close()

    # Restore stdout to the default value
    sys.stdout = sys.__stdout__

    # Plot values Vs Actual
    plt.figure(figsize=(15, 8))
    plt.plot(dates_comparsion, predictions_HL, c="r", label="Predictions", marker="+")
    plt.plot(dates_comparsion, Actual_HL, c="g", label="Actual", marker="^")
    plt.title(stock)
    plt.legend()
    plt.tight_layout()
    ifexists = cd + "\\Imagies" + stockLocation + 'HL Actual VS Predictions.png'

    # Remove the existing file if it exists
    if os.path.exists(ifexists):
        os.remove(ifexists)

    # Save the plot with the new filename
    plt.savefig(cd + "\\Imagies" + stockLocation + 'HL Actual VS Predictions.png')


    #_______________________________________Predicting Future___________________________________________________
    # Get the NYSE calendar
    nyse = get_calendar('NYSE')

    # Get today's date
    today = datetime.now().date()

    # Find the next trading day
    next_trading_day = nyse.valid_days(start_date=today, end_date=today + timedelta(days=30))[0]

    # Get the next five trading days
    next_days = nyse.valid_days(start_date=next_trading_day, end_date=next_trading_day + timedelta(days=15))

    # holds first day and reshapes it and predcits next day
    p_close = y_test_close[-1]
    p_close = scaler_y_close.inverse_transform(p_close.reshape(-1, 1))

    # Array for predictions
    predictions_of_next_days = []

    # Redirecting output to a file
    sys.stdout = open(cd + "\\PredictionTestSet" + stockLocation + 'FuturePredictions.txt', 'w')

    # loop through all days and predict continusly 
    for day in next_days:

        # Obtains year
        year = day.year 
        
        # Make a dataframe to scale dataframe and obatin prdiction and move it to open of next day

        df_holder_HL= pd.DataFrame({'Open': [p_close],'Down': [0],'year': [year]})

        # Scale dataframe of high and low
        holder_day_HL = scaler_X_HL.transform(df_holder_HL)

        # reshape for model high and low
        holder_day_HL = np.reshape(holder_day_HL, (holder_day_HL.shape[0], 1, holder_day_HL.shape[1]))
        
        # Predict High and low from day and inverse scaler
        predict_H = modelHL.predict(holder_day_HL)
        p_HL = scaler_y_HL.inverse_transform(predict_H.reshape(-1, 2))

        # make close model dataframe
        df_holder_close = pd.DataFrame({'Open': [p_close], 'High': [p_HL[0][0]], 'Low': [p_HL[0][1]], 'Down': [0],'year': [year]})

        # Scale dataframe of close
        holder_day_close = scaler_X_close.transform(df_holder_close)

        # reshape for model close
        holder_day_close = np.reshape(holder_day_close, (holder_day_close.shape[0], 1, holder_day_close.shape[1]))

        #Predict close
        predict_close = modelClose.predict(holder_day_close)

        # Inverse scaler with scale y
        p_close = scaler_y_close.inverse_transform(predict_close.reshape(-1, 1))
        
        # append to prediction array
        predictions_of_next_days.append(p_close[0][0])

        # printing out predictions and save to file
        print(str(day) + " Predcition: " + str(p_close[0][0]))
    
    
    # Close the file
    sys.stdout.close()

    # Restore stdout to the default value
    sys.stdout = sys.__stdout__

    # make sure next days are formate for plot
    next_days = pd.to_datetime([day.strftime('%Y-%m-%d') for day in next_days])

    # Plot Test set values to predictions of future
    plt.figure(figsize=(15, 8))
    plt.plot(next_days, predictions_of_next_days, c="r", label="Predictions", marker="+")
    plt.plot(dates_comparsion[size_graph:-1], Actual_close[size_graph:-1], c="g", label="Actual", marker="^")
    plt.title(stock)
    plt.legend()
    plt.tight_layout()
    ifexists = cd + "\\Imagies" + stockLocation + 'Actual VS Predictions Future.png'

    # Remove the existing file if it exists
    if os.path.exists(ifexists):
        os.remove(ifexists)

    # Save the plot with the new filename
    plt.savefig(cd + "\\Imagies" + stockLocation + 'Actual VS Predictions Future.png')

 #_______________________________________Read CSV Data___________________________________________________

def main():

    # AMD = .02 AMZN = .03 NVDA = .035 TSLA = .06
    stocks = ["AMD",'HST',"NVDA","TSLA"]
    test_set_sizes = [0.02,0.03,0.01,0.1]
    batch_size = [32,32,32,64]
    sizes = [100,100,10,75]
    # directory path
    cd = os.getcwd()

    for stock,size in zip(stocks,sizes):
        
        stockLocation = "\\" + stock + "\\"

        predict(cd, stock, stockLocation,size)

# Check if the script is being run as the main module
if __name__ == "__main__":
    # Call the main function
    main()