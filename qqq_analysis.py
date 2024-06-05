import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
start_date = '2014-01-01'
end_date = '2023-12-31'
qqq_data = yf.download('QQQ', start=start_date, end=end_date)
qqq_data.sort_index(inplace=True)
print("Data covers from", qqq_data.index.min(), "to", qqq_data.index.max())


qqq_data['Monthly Return'] = qqq_data['Adj Close'].pct_change().resample('M').last()
qqq_data.dropna(inplace=True)
print(qqq_data.head())
print(qqq_data.tail())

def backtest_dca(data, monthly_investment=500):
    shares_bought = monthly_investment / data['Adj Close']
    cumulative_shares = shares_bought.cumsum()
    total_investment = monthly_investment * len(data)
    portfolio_value = cumulative_shares * data['Adj Close']
    total_return = portfolio_value.iloc[-1] - total_investment
    annualized_return = (portfolio_value.iloc[-1] / total_investment) ** (1 / (len(data)/12)) - 1
    return annualized_return, total_return, portfolio_value

annualized_return_dca, total_return_dca, portfolio_value_dca = backtest_dca(qqq_data)
print(f"DCA Strategy - Annualized Return: {annualized_return_dca:.2%}, Total Return: ${total_return_dca:.2f}")
def momentum_strategy(data, monthly_investment=500):
    momentum = data['Monthly Return'].rolling(window=6).apply(lambda x: x[:-1].sum(), raw=False)
    positions = (momentum > 0).astype(int)
    cash = monthly_investment * (1 - positions)
    investment = monthly_investment * positions
    shares_bought = investment / data['Adj Close']
    cumulative_shares = shares_bought.cumsum()
    portfolio_value = cumulative_shares * data['Adj Close']
    total_investment = monthly_investment * len(data)
    total_return = portfolio_value.iloc[-1] - total_investment
    annualized_return = (portfolio_value.iloc[-1] / total_investment) ** (1 / (len(data)/12)) - 1
    return annualized_return, total_return, portfolio_value


annualized_return_momentum, total_return_momentum, portfolio_value_momentum = momentum_strategy(qqq_data)
print(f"Momentum Strategy - Annualized Return: {annualized_return_momentum:.2%}, Total Return: ${total_return_momentum:.2f}")

def calculate_risk_metrics(data, portfolio_value):
    drawdown = (portfolio_value / portfolio_value.cummax()) - 1
    max_drawdown = drawdown.min()
    volatility = data['Monthly Return'].std() * (12 ** 0.5)
    sharpe_ratio = (data['Monthly Return'].mean() / data['Monthly Return'].std()) * (12 ** 0.5)
    return max_drawdown, volatility, sharpe_ratio
max_drawdown, volatility, sharpe_ratio = calculate_risk_metrics(qqq_data, portfolio_value_dca)

print(f"Risk Metrics - Max Drawdown: {max_drawdown:.2%}, Volatility: {volatility:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}")
def plot_performance(data, dca_value, momentum_value):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, dca_value, label='DCA Strategy')
    plt.plot(data.index, momentum_value, label='Momentum Strategy')
    plt.title('QQQ Investment Strategies Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()
plot_performance(qqq_data, portfolio_value_dca, portfolio_value_momentum)


