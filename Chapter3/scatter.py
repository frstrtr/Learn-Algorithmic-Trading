import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas_datareader import data

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def load_financial_data(start_date, end_date, output_file):
    try:
        df = pd.read_pickle(output_file)
        print('File data found...reading ETHBTC data')
    except FileNotFoundError:
        print('File not found...downloading the ETHBTC data')
        df = data.DataReader('ETHBTC', 'yahoo', start_date, end_date)
        df.to_pickle(output_file)
    return df


def create_regression_trading_condition(df):
    df['Open-Close'] = df.open - df.close
    df['High-Low'] = df.high - df.low
    df['Target'] = df['close'].shift(-1) - df['close']
    df = df.dropna()
    X = df[['Open-Close', 'High-Low']]
    Y = df[['Target']]
    return (df, X, Y)


def create_train_split_group(X, Y, split_ratio=0.8):
    return train_test_split(X, Y, shuffle=False, train_size=split_ratio)


ETHBTC_data = load_financial_data(
    start_date='2001-01-01',
    end_date='2018-01-01',
    output_file='nicehash_ETHBTC_data_days.pk1')

ETHBTC_data, X, Y = create_regression_trading_condition(ETHBTC_data)

pd.plotting.scatter_matrix(
    ETHBTC_data[['Open-Close', 'High-Low', 'Target']], grid=True, diagonal='kde', alpha=0.8)

plt.show()

X_train, X_test, Y_train, Y_test = create_train_split_group(
    X, Y, split_ratio=0.8)


ols = linear_model.LinearRegression()
ols.fit(X_train, Y_train)

print('Coefficients: \n', ols.coef_)

# The mean squared error
print("Mean squared error: %.2f" %
      mean_squared_error(Y_train, ols.predict(X_train)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_train, ols.predict(X_train)))
# The mean squared error
print("Mean squared error: %.2f" %
      mean_squared_error(Y_test, ols.predict(X_test)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, ols.predict(X_test)))

ETHBTC_data['Predicted_Signal'] = ols.predict(X)
ETHBTC_data['ETHBTC_Returns'] = np.log(ETHBTC_data['close'] /
                                       ETHBTC_data['close'].shift(1))


def calculate_return(df, split_value, symbol):
    cum_ETHBTC_return = df[split_value:]['%s_Returns' % symbol].cumsum() * 100
    df['Strategy_Returns'] = df['%s_Returns' %
                                symbol] * df['Predicted_Signal'].shift(1)
    return cum_ETHBTC_return


def calculate_strategy_return(df, split_value, symbol):
    cum_strategy_return = df[split_value:]['Strategy_Returns'].cumsum() * 100
    return cum_strategy_return


cum_ETHBTC_return = calculate_return(
    ETHBTC_data, split_value=len(X_train), symbol='ETHBTC')
cum_strategy_return = calculate_strategy_return(ETHBTC_data,
                                                split_value=len(X_train), symbol='ETHBTC')


def plot_chart(cum_symbol_return, cum_strategy_return, symbol):
    plt.figure(figsize=(10, 5))
    plt.plot(cum_symbol_return, label='%s Returns' % symbol)
    plt.plot(cum_strategy_return, label='Strategy Returns')
    plt.legend()


plot_chart(cum_ETHBTC_return, cum_strategy_return, symbol='ETHBTC')


def sharpe_ratio(symbol_returns, strategy_returns):
    strategy_std = strategy_returns.std()
    sharpe = (strategy_returns - symbol_returns) / strategy_std
    return sharpe.mean()


print(sharpe_ratio(cum_strategy_return, cum_ETHBTC_return))
