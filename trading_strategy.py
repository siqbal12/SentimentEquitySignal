from ml_modeling import *
import matplotlib.pyplot as plt

backtest = X_test.copy()

# backtest['Return_tomorrow'] = df_test['Return_tomorrow']

backtest['prob_up'] = ml_model.predict_proba(backtest)[:, 1]
backtest['position'] = (backtest['prob_up'] > 0.5).astype(int)
backtest['strategy_return'] = backtest['position'] * df_test['Return_tomorrow']
backtest['buy_hold_return'] = df_test['Return_tomorrow']

backtest['date'] = df_test['date']

daily_strategy_return = (
    backtest.groupby('date')['strategy_return']
            .mean()
)
daily_buy_hold_return = (
    backtest.groupby('date')['buy_hold_return']
            .mean()
)

strategy_cumulative_return = (
    1 + daily_strategy_return
).cumprod()

buy_hold_cumulative_return = (
    1 + daily_buy_hold_return
).cumprod()


# backtest['strategy_cumulative_return'] = (1 + backtest['strategy_return']).cumprod()
# backtest['buy_hold_cumulative_return'] = (1 + backtest['buy_hold_return']).cumprod()

#Sharpes
strategy_sharpe = (
        daily_strategy_return.mean() / daily_strategy_return.std()
    ) * np.sqrt(252)
buy_hold_sharpe = (
        daily_buy_hold_return.mean() / daily_buy_hold_return.std()
    ) * np.sqrt(252)

print('Strategy Sharpe:, ', strategy_sharpe)
print('Buy and Hold Sharpe:, ', buy_hold_sharpe)

#Max Drawdown
strategy_running_max = strategy_cumulative_return.cummax()
strategy_drawdown = (strategy_cumulative_return - strategy_running_max) / strategy_running_max
strategy_max_drawdown = strategy_drawdown.min()

buy_hold_running_max = buy_hold_cumulative_return.cummax()
buy_hold_drawdown = (buy_hold_cumulative_return - buy_hold_running_max) / buy_hold_running_max
buy_hold_max_drawdown = buy_hold_drawdown.min()

print('Strategy Max DD:, ', strategy_max_drawdown)
print('Buy and Hold Max DD:, ', buy_hold_max_drawdown)

#Total Return
strategy_total_return = strategy_cumulative_return.iloc[-1] - 1
buy_hold_total_return = buy_hold_cumulative_return.iloc[-1] - 1

print('Strategy Total Return:, ', strategy_total_return)
print('Buy and Hold Total Return:, ', buy_hold_total_return)

#Plots
plt.figure()
plt.plot(strategy_cumulative_return.index, strategy_cumulative_return, color='green', label='Strategy')
plt.plot(strategy_cumulative_return.index, buy_hold_cumulative_return, color='blue', label='Buy and Hold')
plt.plot(strategy_cumulative_return.index, [1] * len(strategy_cumulative_return.index), label='Baseline', color='red', ls='--')
plt.title(f"Cumulative Return")
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.xticks(strategy_cumulative_return.index[::10],rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


daily_no_best = daily_strategy_return.drop(
    daily_strategy_return.idxmax()
)

cum_no_best = (1 + daily_no_best).cumprod()

sharpe_no_best = (
    daily_no_best.mean()
    /
    daily_no_best.std()
) * np.sqrt(252)

print('Strategy Sharpe No Best: ', sharpe_no_best)
print('Strategy No Best Total Return: ', cum_no_best.iloc[-1] - 1)


x = True