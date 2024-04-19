# Imports from Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


#______________________________________Description of Data____________________________________________________


def Description(df, cd, stockLocation,stock):

    """
    Print the stock info:
        Prints Dataframe of stock

        Prints describtion of data:
            Describe: what each value is like Intger or Float and if any nulls
        Prints info:
            Count: count of how many rows inside
            Mean: mean of columns
            std: standard deviation of all columns
            min: min value of each column
            25%: 25% quartile of values
            50%: 50% Quartile of values
            75%: 75% Quartile of values
            max: max value of columns

        variance:
            print variance of all columns

    """

    # Redirecting output to a file
    sys.stdout = open(cd + "\\Stats" + stockLocation + 'Stats.txt', 'w')

    # Dataframe printed
    print("DATAFRAME")
    print(df)

    # Decribtion of Data
    print("\n DESCRIBTION")
    print(df.describe())

    # Info of Data
    print("\n INFO")
    print(df.info())

    # Variance of columns
    print("\n VARIANCE")
    variance_per_column = df.var()
    print("Variance per column:\n", variance_per_column)

    # Close the file
    sys.stdout.close()

    # Restore stdout to the default value
    sys.stdout = sys.__stdout__


#___________________________________Correlation Heatmap_______________________________________________________

def Heatmap(df, cd, stockLocation, stock):
    """
    Corelation Heatmap:
            Print correlation matrix
            Next call plots it and saves imagie
    """
    # Here to since if could exist speeds up process
    ifExists = cd + "\\Imagies" + stockLocation + 'correlation Heatmap.png'

    # obtains correlation matrix
    correlation_matrix = df.corr()
    print("Correlation Matrix:")
    print(correlation_matrix)

    # If not make heatmap
    if not os.path.exists(ifExists):

        # Plots correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(stock + ' Correlation Heatmap')
        plt.savefig(cd + "\\Imagies" + stockLocation + 'correlation Heatmap.png')

#_______________________________________Graphs___________________________________________________

def plots(df, cd, stockLocation, stock):
    """
    Plots graphs:
        1st: Open vs Close: simliarity and increases
        2nd: Volume vs Date: Shows populatory
        3rd: Date VS close to see data
    """
    # Here to since if could exist speeds up process
    ifExists = cd + "\\Imagies" + stockLocation + 'Close VS Open.png'

    # If one plot exist they should all exist already
    if not os.path.exists(ifExists):

        # Open Vs Close
        plt.figure(figsize=(15, 8))
        plt.plot(df.index, df['Close'], c="r", label="Close", marker="+")
        plt.plot(df.index, df['Open'], c="g", label="Open", marker="^")
        plt.title(stock)
        plt.legend()
        plt.xticks(df.index[::250], df['year'][::250])
        plt.tight_layout()
        plt.savefig(cd + "\\Imagies" + stockLocation +'Close VS Open.png')

        # date vs volume
        plt.figure(figsize=(15, 8))
        plt.plot(df.index, df['Volume'], c='purple', marker='*')
        plt.title(stock +" Volume")
        plt.xticks(df.index[::250], df['year'][::250])
        plt.tight_layout()
        plt.savefig(cd + "\\Imagies" + stockLocation +'Volume VS Close.png')

        # date vs close
        plt.figure(figsize=(15, 8))
        plt.plot(df.index, df['Close'])
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.title(stock + "Stock Prices")
        plt.xticks(df.index[::250], df['year'][::250])
        plt.tight_layout()
        plt.savefig(cd + "\\Imagies" + stockLocation +'Date VS Close.png')