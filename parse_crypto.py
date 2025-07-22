import pandas as pd
import io
import os
import requests

if __name__ == '__main__':

    try:
        os.mkdir("data")
    except:
        pass

    link = "https://data.bitcoinity.org/export_data.csv?currency=USD&data_type=price_volume&r=day&t=lb&timespan=2y&vu=curr"
    file = requests.get(link)
    df = pd.read_csv(io.BytesIO(file.content))

    df['date'] = pd.to_datetime(df['Time']).dt.date
    df.rename(columns={'volume': 'number_sold'}, inplace=True)
    df = df[['date', 'price', 'number_sold']]

    df.to_csv("data/BTC.csv", index=False)