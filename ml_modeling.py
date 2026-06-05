import pandas as pd
import re
from itertools import chain
from collections import Counter
import yfinance as yf
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, average_precision_score

df_wsb_aggregated = pd.read_csv('df_wsb_aggregated.csv')
df_train = df_wsb_aggregated.loc[df_wsb_aggregated['date'] < '2021-04-01', :]
df_validation = df_wsb_aggregated.loc[
           (df_wsb_aggregated['date'] >= '2021-04-01') & (df_wsb_aggregated['date'] < '2021-06-01'),
           :]
df_test = df_wsb_aggregated.loc[df_wsb_aggregated['date'] >= '2021-06-01', :]

MARKET_COLUMNS = ['Return_1d', 'Return_5d', 'Realized_volatility_5d', 'Log_Volume']
SENTIMENT_COLUMNS = ['positive_avg', 'positive_std',
                     'negative_avg', 'negative_std',
                     'neutral_avg', 'neutral_std',
                     'sentiment_score_avg', 'sentiment_score_std',
                     'mention_count']

LINEAR_TARGET = 'Return_tomorrow'
CLASSIFICATION_TARGET = 'Is_Return_tomorrow_positive'

results_dict = {'dataset_type': [], 'ml_model_type': [], 'output_type': [],
                'MSE': [], 'R2': [],
                'Accuracy': [], 'ROC AUC': [], 'PR AUC': []}

for dataset_type in ['Market Only', 'Sentiment Only', 'Market & Sentiment', 'Mention', 'Mention & Sentiment Std']:
    if dataset_type == 'Market Only':
        X_columns = MARKET_COLUMNS
    elif dataset_type == 'Sentiment Only':
        X_columns = SENTIMENT_COLUMNS
    elif dataset_type == 'Market & Sentiment':
        X_columns = MARKET_COLUMNS + SENTIMENT_COLUMNS
    elif dataset_type == 'Mention':
        X_columns = ['mention_count']
    elif dataset_type == 'Mention & Sentiment Std':
        X_columns = ['mention_count', 'sentiment_score_std']

    X_train = df_train.loc[:, X_columns]
    X_validation = df_validation.loc[:, X_columns]
    X_test = df_test.loc[:, X_columns]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    for ml_model_type in ['Linear', 'Random Forest', 'XGBoost']:

        for output_type in ['Regression', 'Classification']:

            if ml_model_type == 'Linear':

                if output_type == 'Regression':
                    ml_model = LinearRegression()
                else:
                    ml_model = LogisticRegression(penalty=None)

            elif ml_model_type == 'Random Forest':
                if output_type == 'Regression':
                    ml_model = RandomForestRegressor()
                else:
                    ml_model = RandomForestClassifier()

            elif ml_model_type == 'XGBoost':
                if output_type == 'Regression':
                    ml_model = XGBRegressor()
                else:
                    ml_model = XGBClassifier()

            y_column = LINEAR_TARGET if output_type == 'Regression' else CLASSIFICATION_TARGET

            y_train = df_train.loc[:, y_column]
            y_validation = df_validation.loc[:, y_column]
            y_test = df_test.loc[:, y_column]

            ml_model.fit(X_train, y_train) if ml_model_type != 'Linear' else ml_model.fit(X_train_scaled, y_train)

            if output_type == 'Regression':
                y_validation_pred = ml_model.predict(X_validation) if ml_model_type != 'Linear' else ml_model.predict(X_validation_scaled)
                mse = mean_squared_error(y_validation, y_validation_pred)
                r2 = r2_score(y_validation, y_validation_pred)
                accuracy = None
                roc_auc = None
                pr_auc = None


            if output_type == 'Classification':
                y_validation_pred = ml_model.predict(X_validation) if ml_model_type != 'Linear' else ml_model.predict(X_validation_scaled)
                y_validation_pred_proba = ml_model.predict_proba(X_validation)[:, 1] if ml_model_type != 'Linear' else ml_model.predict_proba(X_validation_scaled)[:, 1]
                # if 'Sentiment' in dataset_type:
                #     y_validation_pred_proba = 1 - y_validation_pred_proba
                # y_validation_pred_proba = y_validation_pred_proba[:, 0] if 'Sentiment' in dataset_type else y_validation_pred_proba[:, 1]
                mse = None
                r2 = None
                accuracy = accuracy_score(y_validation, y_validation_pred)
                roc_auc = roc_auc_score(y_validation, y_validation_pred_proba)
                pr_auc = average_precision_score(y_validation, y_validation_pred_proba)

            results_dict['dataset_type'].append(dataset_type)
            results_dict['ml_model_type'].append(ml_model_type)
            results_dict['output_type'].append(output_type)
            results_dict['MSE'].append(mse)
            results_dict['R2'].append(r2)
            results_dict['Accuracy'].append(accuracy)
            results_dict['ROC AUC'].append(roc_auc)
            results_dict['PR AUC'].append(pr_auc)

df_results = pd.DataFrame(results_dict)


#We will choose:
# - Market & Sentiment Data
# - Logistic Regression

scaler = StandardScaler()
X_train = df_train.loc[:, MARKET_COLUMNS + SENTIMENT_COLUMNS]
X_test = df_test.loc[:, MARKET_COLUMNS + SENTIMENT_COLUMNS]
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = df_train.loc[:, CLASSIFICATION_TARGET]
y_test = df_test.loc[:, CLASSIFICATION_TARGET]

ml_model = LogisticRegression(penalty=None)
ml_model.fit(X_train_scaled, y_train)







x = True