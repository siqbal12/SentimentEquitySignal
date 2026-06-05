Sentiment-Based Signal Generation from Reddit Data
•	Alternative Data: Processed 100k+ WallStreetBets posts using FinBERT sentiment analysis and attention-based features; trained Logistic Regression, Random Forest, and XGBoost models to forecast next-day meme-stock returns
•	Signal Construction & Evaluation: Translated ML predictions to signals of expected positive returns; dynamically timed equity exposure to increase Sharpe (0.95 to 1.70) and reduce Drawdown (40% to 26%) from classic buy-and-hold strategy

To run on your own:
- Download the reddit data from https://www.kaggle.com/datasets/leukipp/reddit-finance-data
- To process the data
    - Run api_playground.py
- To aggregate the data (by date and ticker)
    - Run data_aggregations.py
- To apply the NLP Sentiment Analysis
    - Run sentiment_analysis.py
- To create the ML models
    - Run ml_modeling.py
- To run the trading strategy
    - Run trading_strategy.py
