import pandas as pd

#________________________________________Adding Extra data__________________________________________________

def extras(df):
    """"
    Adding of extra data to stock:
        Up Variable: Saying it went up 0=no 1=yes
        Down Variable: Saying it went down 0=no 1=yes
        Day Variable: Day
        Month Variable: Month
        Year Variable: Year
    """

    # adding extras to dataframe
    df['Up'] = (df['Close'] > df['Open']).astype(int)
    df['Down'] = (df['Close'] < df['Open']).astype(int)
    df['Date'] = pd.to_datetime(df['Date'])
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year

    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Set the date to index 
    df = df.set_index('Date')

    # Filter if you want only certain dates
    #df = df[df.index >= '2018-01-01']

    return df
