import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys
import pandas as pd
from pandas_market_calendars import get_calendar
import pandas as pd
from pandas_market_calendars import get_calendar
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error


#_______________________________________Load___________________________________________________
def predict(cd,stock,stockLocation,size_graph):

    # Load the trained model
    model = joblib.load(cd + "\\Model\\" + stockLocation + "sklearnModel.joblib")


    # Load the test set features and labels from numpy arrays
    X_test = np.load(cd + "\\npy\\" + stockLocation +"X_test.npy")
    y_test = np.load(cd + "\\npy\\" + stockLocation +"y_test.npy")
    dates = np.load(cd + "\\npy\\" + stockLocation +"dates.npy")

    #_______________________________________Predicting Test for accuracy___________________________________________________

    # Load the scaler object for y,x for predicting
    scaler_y = joblib.load(cd + "\\Scaler\\" +stockLocation +"scaler_y.pkl")
    scaler_X = joblib.load(cd + "\\Scaler\\" +stockLocation +"scaler_X.pkl")

    # Evaluate
    regressionscore = model.score(X_test, y_test)

    # Prediction
    predictions = model.predict(X_test)

    # Inverse scaler for prediction
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))

    # Scalers the actual back
    Actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # Used for later to only compare dates in test set
    dates_comparsion = []

    # Redirecting output to a file
    sys.stdout = open(cd + "\\PredictionTestSet" + stockLocation + 'TestPredictions.txt', 'w')

    # Adds the correct dates from the dates npy file so we have correct dates
    for i in range(y_test.size):
        # Prints date 
        print("\n", dates[i])
        print
        dates_comparsion.append(dates[i])
        print("Prediction: ", predictions[i][0])  
        print("Actual: ", Actual[i][0]) 

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Test accuracy
    print(regressionscore)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", np.sqrt(mse))

    # Close the file
    sys.stdout.close()

    # Restore stdout to the default value
    sys.stdout = sys.__stdout__

    # Plot values Vs Actual
    plt.figure(figsize=(15, 8))
    plt.plot(dates_comparsion, predictions, c="r", label="Predictions", marker="+")
    plt.plot(dates_comparsion, Actual, c="g", label="Actual", marker="^")
    plt.title(stock)
    plt.legend()
    plt.tight_layout()
    ifexists = cd + "\\Imagies" + stockLocation + 'Actual VS Predictions.png'

    # Remove the existing file if it exists
    if os.path.exists(ifexists):
        os.remove(ifexists)

    # Save the plot with the new filename
    plt.savefig(cd + "\\Imagies" + stockLocation + 'Actual VS Predictions.png')


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
    p = y_test[-1]

    p = scaler_y.inverse_transform(p.reshape(-1, 1))

    # Array for predictions
    predictions_of_next_days = []

    # Redirecting output to a file
    sys.stdout = open(cd + "\\PredictionTestSet" + stockLocation + 'FuturePredictions.txt', 'w')

    # loop through all days and predict continusly 
    for day in next_days:

        # Obtains year
        year = day.year 
        
        # Make a dataframe to scale dataframe and obatin prdiction and move it to open of next day
        df_holder = pd.DataFrame({'Open': [p],'year': [year]})

        # Scale dataframe
        holder_day = scaler_X.transform(df_holder)

        # reshape for model
        holder_day = holder_day.reshape(1, -1)
        
        # Predict day
        predict = model.predict(holder_day)
        
        # Inverse scaler with scale y
        p = scaler_y.inverse_transform(predict.reshape(-1, 1))
        
        # append to prediction array
        predictions_of_next_days.append(p[0][0])

        # printing out predictions and save to file
        print(str(day) + " Predcition: " + str(p[0][0]))
    
    
    # Close the file
    sys.stdout.close()

    # Restore stdout to the default value
    sys.stdout = sys.__stdout__

    # make sure next days are formate for plot
    next_days = pd.to_datetime([day.strftime('%Y-%m-%d') for day in next_days])

    # Plot Test set values to predictions of future
    plt.figure(figsize=(15, 8))
    plt.plot(next_days, predictions_of_next_days, c="r", label="Predictions", marker="+")
    plt.plot(dates_comparsion[size_graph:-1], Actual[size_graph:-1], c="g", label="Actual", marker="^")
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