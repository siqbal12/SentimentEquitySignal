import pandas as pd
import re
from itertools import chain
from collections import Counter
import yfinance as yf
import numpy as np

df_wsb = pd.read_csv('df_wsb_sentiment.csv')
df_wsb['date'] = pd.to_datetime(df_wsb['date']).dt.date

df_wsb_aggregated = df_wsb.groupby(['date', 'ticker']).agg(
    positive_avg = ('positive', 'mean'),
    positive_std = ('positive', 'std'),
    negative_avg = ('negative', 'mean'),
    negative_std = ('negative', 'std'),
    neutral_avg = ('neutral', 'mean'),
    neutral_std = ('neutral', 'std'),
    sentiment_score_avg = ('sentiment_score', 'mean'),
    sentiment_score_std = ('sentiment_score', 'std'),
    mention_count=('ticker', 'count')
).reset_index()

df_wsb_aggregated = df_wsb_aggregated.loc[df_wsb_aggregated['mention_count'] >= 20, :]

def get_ticker_prices(ticker):

    df_prices = yf.download(ticker, start='2020-12-01', end='2022-01-01', auto_adjust=True, progress=False)
    df_prices.columns =['Close', 'High', 'Low', 'Open', 'Volume']
    df_prices['ticker'] = ticker
    df_prices = df_prices.reset_index().loc[:, ['Date', 'ticker', 'Close', 'Volume']].rename(columns={'Date': 'date'})
    df_prices['date'] = pd.to_datetime(df_prices['date']).dt.date
    df_prices['Return_1d'] = df_prices['Close'].pct_change()
    df_prices['Return_5d'] = df_prices['Close'] / df_prices['Close'].shift(5) - 1
    df_prices['Return_tomorrow'] = df_prices['Close'].shift(-1) / df_prices['Close'] - 1
    df_prices['Realized_volatility_5d'] = df_prices['Return_1d'].rolling(5).std()
    df_prices['Log_Volume'] = np.log1p(df_prices['Volume'])

    df_prices['Is_Return_tomorrow_positive'] = (df_prices['Return_tomorrow'] > 0).astype(int)

    return df_prices

df_gme_prices = get_ticker_prices('GME')
df_amc_prices = get_ticker_prices('AMC')
df_bb_prices = get_ticker_prices('BB')
df_nok_prices = get_ticker_prices('NOK')
df_prices = pd.concat([df_gme_prices, df_amc_prices, df_bb_prices, df_nok_prices ])

df_wsb_aggregated = pd.merge(df_wsb_aggregated, df_prices, on=['date', 'ticker'], how='left')
df_wsb_aggregated = df_wsb_aggregated.dropna()

df_wsb_aggregated.to_csv('df_wsb_aggregated.csv')

x = True