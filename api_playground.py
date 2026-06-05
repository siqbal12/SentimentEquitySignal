import pandas as pd
import re
from itertools import chain
from collections import Counter

keywords = ['GME', 'AMC', 'BB', 'NOK',]

subreddits = ['wallstreetbets',
              # 'stocks',
              # 'stockmarket',
              # 'robinhoodpennystocks',
              # 'robinhood',
              # 'personalfinance',
              # 'pennystocks',
              # 'options',
              # 'investing'
              ]

matches_flat_total = []
df_dict = dict()
for subreddit in subreddits:
    print(subreddit)
    file_name = f'{subreddit}_reddit.csv'
    df = pd.read_csv(file_name)

    titles = list(df['title'])
    matches = [re.findall(r'\b[A-Z]{1,5}\b', str(title)) for title in titles]
    df['tickers'] = matches
    matches_flat = list(chain.from_iterable(matches))
    matches_flat_total += matches_flat

    s_bool = [any(keyword in match for keyword in keywords) for match in matches]
    df = df.loc[s_bool, :]

    df_dict[subreddit] = df

def extract_specific_ticker(ticker_list):
    if 'GME' in ticker_list:
        return 'GME'
    elif 'AMC' in ticker_list:
        return 'AMC'
    elif 'BB' in ticker_list:
        return 'BB'
    else:
        return 'NOK'

df_wsb = df_dict['wallstreetbets']
df_wsb['ticker'] = df_wsb['tickers'].apply(lambda ticker_list: extract_specific_ticker(ticker_list))
del df_wsb['tickers']

RELEVANT_COLUMNS = ['created',
                    'ticker',
                    'title', 'selftext',
                    'upvote_ratio', 'score', 'gilded', 'total_awards_received', 'num_comments', 'num_crossposts',
                    ]

df_wsb = df_wsb.loc[:, RELEVANT_COLUMNS].rename(columns={'created': 'date'})
df_wsb.to_csv('df_wsb.csv')

x = True