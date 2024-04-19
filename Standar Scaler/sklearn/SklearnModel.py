# Imports from Libraries
from sklearn.linear_model import LinearRegression
import joblib

#_____________________________________Model LSTM_____________________________________________________

def modelTrain(X_train,y_train, stockLocation, cd):

    regr = LinearRegression()
 
    regr.fit(X_train, y_train)
        
    model_filename = "sklearnModel.joblib"
    joblib.dump(regr, cd + "/Model/" + stockLocation + model_filename)
    print(f"Model saved to {model_filename}")

    