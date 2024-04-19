# Imports from Libraries
from keras.models import Sequential
from keras.layers import LSTM,Dense
import tensorflow as tf


#_____________________________________Model LSTM_____________________________________________________

def modelTrain(X_train,y_train, stockLocation, cd, batch,model_type,shape):
    
    # Making of model and layers
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(16, activation='relu'),
        Dense(shape)
    ])

    # THe optimizer for the model: using Adam wiht learning rate 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0)


    # Compile the model
    model.compile(optimizer= optimizer,
            loss='mse',  # Change the loss function to mean squared error
            metrics=['mae', 'mse']
            )

    # Fits the model with the data/Train sets
    history = model.fit(X_train, y_train, epochs=100, batch_size=batch, verbose=1,shuffle= False)

    # Save the trained model
    model.save(cd + "\\Model\\" + stockLocation + model_type + "TrainedLSTM.h5")


    