import CleaningAndSets as Clean
import ModelLSTM as M
import Graphs_Stats as Stats
import os
import Predict_Test as Pre
import ReadCSV as R
import AddExtras as Extra

 #_______________________________________Read CSV Data___________________________________________________

def main():

    # AMD = .02 AMZN = .03 NVDA = .035 TSLA = .06
    stocks = ["AMD",'HST',"NVDA","TSLA"]
    test_set_sizes = [0.02,0.03,0.01,0.1]
    batch_size = [32,32,32,64]
    sizes = [100,100,10,75]

    # directory path
    cd = os.getcwd()

    for stock, set_size, batch,size in zip(stocks, test_set_sizes,batch_size,sizes):
        
        stockLocation = "\\" + stock + "\\"

        data = R.read(stock,cd)

        data = Extra.extras(data)

        Stats.Description(data,cd,stockLocation,stock)

        Stats.Heatmap(data,cd,stockLocation,stock)

        Stats.plots(data,cd,stockLocation,stock)

        X_train, y_train = Clean.cleaning(data,cd,stockLocation,set_size)

        M.modelTrain(X_train,y_train,stockLocation,cd,batch)

        Pre.predict(cd,stock,stockLocation,size)

# Check if the script is being run as the main module
if __name__ == "__main__":
    # Call the main function
    main()
