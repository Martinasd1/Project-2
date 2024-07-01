import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    return data['Adj Close']
def calculate_ma(data, window):
    return data.rolling(window=window).mean()
def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ma_rsi_strategy(data, ma_window, rsi_window, rsi_buy_threshold, rsi_sell_threshold):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data
    signals['ma'] = calculate_ma(data, ma_window)
    signals['rsi'] = calculate_rsi(data, rsi_window)
    signals['signal'] = 0
    
 
    signals.loc[(signals['price'] > signals['ma']) & (signals['rsi'] < rsi_buy_threshold), 'signal'] = 1
    signals.loc[(signals['price'] < signals['ma']) & (signals['rsi'] > rsi_sell_threshold), 'signal'] = -1

    return signals

def backtest_strategy(signals, initial_capital=10000):
    positions = signals['signal'].diff()
    cash = initial_capital
    holdings = 0
    total_value = []
    
    for date, signal in signals.iterrows():
        if signal['signal'] == 1 and cash > 0:
            holdings = cash / signal['price']
            cash = 0
        elif signal['signal'] == -1 and holdings > 0:
            cash = holdings * signal['price']
            holdings = 0
        total_value.append(cash + holdings * signal['price'])
    
    signals['total_value'] = total_value
    return signals

def plot_ma_rsi_results(backtest_results, benchmark):
    plt.figure(figsize=(14, 7))
    for ticker in backtest_results:
        plt.plot(backtest_results[ticker].index, backtest_results[ticker]['total_value'], label=ticker)
    plt.plot(benchmark.index, benchmark['total_value'], label='SP500', linestyle='--')
    plt.legend()
    plt.title('MA-RSI Strategy Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.show()


def plot_performance_metrics(performance_metrics):
    performance_metrics.plot(kind='bar', subplots=True, layout=(3, 1), figsize=(14, 14))
    plt.suptitle('Performance Metrics')
    plt.show()


def calculate_performance_metrics(backtest_results):
    performance_metrics = {}
    for ticker in backtest_results:
        total_value = backtest_results[ticker]['total_value']
        returns = total_value.pct_change().dropna()
        annualized_return = (total_value.iloc[-1] / total_value.iloc[0]) ** (1 / (len(total_value) / 252)) - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        max_drawdown = (total_value / total_value.cummax() - 1).min()
        performance_metrics[ticker] = [annualized_return, sharpe_ratio, max_drawdown]
    
    performance_metrics_df = pd.DataFrame(performance_metrics, index=['Annualized Return', 'Sharpe Ratio', 'Max Drawdown'])
    return performance_metrics_df.T

tickers = ['AAPL', 'AMZN', 'NFLX', 'META', 'GOOG', '^GSPC']
start_date = '2014-01-01'
end_date = '2024-01-01'
ma_window = 50
rsi_window = 14
rsi_buy_threshold = 30
rsi_sell_threshold = 70
initial_capital = 10000

data = get_data(tickers, start_date, end_date)

signals = {}
for ticker in tickers[:-1]:  # 排除SP500指数
    signals[ticker] = ma_rsi_strategy(data[ticker], ma_window, rsi_window, rsi_buy_threshold, rsi_sell_threshold)

backtest_results = {}
for ticker in signals:
    backtest_results[ticker] = backtest_strategy(signals[ticker], initial_capital)

benchmark = backtest_strategy(ma_rsi_strategy(data['^GSPC'], ma_window, rsi_window, rsi_buy_threshold, rsi_sell_threshold), initial_capital)

plot_ma_rsi_results(backtest_results, benchmark)

performance_metrics = calculate_performance_metrics(backtest_results)

plot_performance_metrics(performance_metrics)

print(performance_metrics)
