import pandas as pd
import re
from itertools import chain
from collections import Counter

df_wsb = pd.read_csv('df_wsb.csv')

#Finbert
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from transformers.utils import logging
logging.set_verbosity_info()

classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

def get_finbert_scores(text):
    result = classifier(
        str(text),
        top_k=None,
        truncation=True
    )

    scores = {x["label"]: x["score"] for x in result}

    return {
        "positive": scores["positive"],
        "negative": scores["negative"],
        "neutral": scores["neutral"],
        "sentiment_score": (
            scores["positive"] - scores["negative"]
        )
    }

sentiment_df = df_wsb["title"].apply(get_finbert_scores)

sentiment_df = sentiment_df.apply(pd.Series)

df_wsb = pd.concat(
    [df_wsb, sentiment_df],
    axis=1
)

x = True