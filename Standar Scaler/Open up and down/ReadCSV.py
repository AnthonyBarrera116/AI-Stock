import pandas as pd
#_______________________________________Read CSV Data___________________________________________________
def read(stock,cd):

    # Reading stock CSV
    df = pd.read_csv(cd + "\\CSV\\" + stock + ".csv")

    # returns dataframe
    return df