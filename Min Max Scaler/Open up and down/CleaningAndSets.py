# Imports from Libraries
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

#___________________________________________Cleaning and Test Sets_______________________________________________

def cleaning(df,cd,stockLocation,test_set_size):
    """
    Cleaning up the data and adding what values we would like to keep
    This is important if values have very little correlation or even negtive correlation

    """
    # Only take X Open and Year. Y is close
    X, y = df.drop(columns=["Close",'Adj Close','Volume','month','day','High','Low']), df['Close']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size,shuffle=False)

    # Saving dates for later comaprsion of actual and predicted
    index_array = y_test.index.to_numpy(dtype='datetime64')

    # Save the index array
    np.save(cd + "\\npy\\" + stockLocation + "dates.npy", index_array)

    #print(X_train)

    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Implment Standar Scaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Scale X and Y Train sets
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))

    # change X train shape
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    # Scale X and Y test Sets
    X_test = scaler_X.fit_transform(X_test)
    y_test = scaler_y.fit_transform(np.array(y_test).reshape(-1, 1))

    # change X Test shape
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Save the test set features and labels to numpy arrays
    np.save(cd + "\\npy\\" + stockLocation +"X_test.npy", X_test)
    np.save(cd + "\\npy\\" + stockLocation +"y_test.npy", y_test)

    # Save the scaler object
    joblib.dump(scaler_y, cd + "\\Scaler\\" + stockLocation + "scaler_y.pkl")
    joblib.dump(scaler_X, cd + "\\Scaler\\" + stockLocation + "scaler_X.pkl")

    # return train sets to model and Tests are saved for later use in predict
    return  X_train, y_train

