To run you must install: 
GPU USE:
conda install keras-gpu=2.6 numpy=1.22.3 tensorflow-gpu=2.6.0 scipy=1.7.3 scikit-learn=1.3.0 seaborn=0.12.2 plotly=5.19.0 pytorch theano=1.0.5 pandas=2.0.3 cudatoolkit=11.3.1 
pip install matplotlib==3.7.5 

CPU USE:
conda install keras numpy tensorflow scipyscikit-learn seabornplotly pytorch theano pandas
pip install matplotlib pytorch torch torchvision 

Add folder of name of stock in the folders where some stocks are. Add your csv file and change the for loop with your stock name. Once done run what model you would like to test and wait. Imagies will be put in imagies and Predictions and test set predictions are in PredictionTestSet
It will show MAE, MSE, RMSE in text file
