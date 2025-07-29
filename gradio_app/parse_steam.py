import pandas as pd
import re
import json
import os
import time
import requests

PARSE_DEFAULT_TIME_DELAY = 18
PARSE_ERROR_TIME_DELAY = 60

parse_time_out_string = r"You've made too many requests recently. Please wait and try your request again later."
url_head = 'https://steamcommunity.com/market/listings/730/'

def parse_steam_directly(name:str, days:int = None):
    parse_link = f'{url_head}/{name}'
    page = fetch_page(parse_link)
    df = parse_pricing(page)
    df = df[['date', 'price', 'number_sold']]

    if days is not None:
        df = df[-days:]

    return df

def fetch_page(url):
    r = requests.get(url)
    page = r.content.decode('utf8')
    while re.search(parse_time_out_string, page) is not None:
        print('delay')
        time.sleep(PARSE_ERROR_TIME_DELAY)
        r = requests.get(url)
        page = r.content.decode('utf8')
    return page

def parse_pricing(page) -> pd.DataFrame:
    #finds the trade history
    pattern = r"\[\[.*?\]\]"
    match_p = re.search(pattern, page).group(0)[1:-1]
    #creates the dataframe on a created history
    ent_df = pd.DataFrame(re.findall(r"\[.*?\]", match_p), columns=['date']) #findall: divides list on elements

    ent_df[['date', 'price', 'number_sold']] = ent_df['date'].str.split(',', n=2, expand=True)

    ent_df['date'] = ent_df['date'].str[2:-8] #[2:-8], because we are throwing away the bracket(2:) and time(:-8), because it's all the same
    ent_df['date'] = pd.to_datetime(ent_df['date'], format='%b %d %Y')
    ent_df['number_sold'] = ent_df['number_sold'].str[1:-2]# [1:-2] - throwing away brackets
    # changing types
    ent_df['price'] = ent_df['price'].astype('float')
    ent_df['number_sold'] = ent_df['number_sold'].astype('int')
    # throwing away the last day, because it mostly won't be complete
    ent_df = ent_df[ent_df['date'] != ent_df['date'].max()]
    # selecting the last month
    last_month = ent_df[ent_df['date'] > (ent_df["date"].max() - pd.Timedelta(days=31))]
    last_month = last_month.groupby(['date']).agg(
        price=('price', 'mean'),
        number_sold=('number_sold', 'sum')
    ).reset_index()
    # concatenate it back
    ent_df = ent_df[ent_df['date'] <= (ent_df["date"].max() - pd.Timedelta(days=31))]
    ent_df = pd.concat([ent_df, last_month], ignore_index=True)
    print('Successfully parsed pricing data')
    return ent_df

if __name__ == '__main__':
    with open('../parse_list.json', 'r', encoding='utf-8') as f:
        parse_list = json.load(f)

    parse_links = [f'{url_head}/{x}' for x in parse_list]

    try:
        os.mkdir("data")
    except:
        pass

    for name in parse_list:
        df = parse_steam_directly(name)
        name = name.replace(" ", "_")
        name = name.replace("&", "_")
        df.to_csv(f"data/{name}.csv", index=False)