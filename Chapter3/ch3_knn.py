import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


from pandas_datareader import data
start_date = '2001-01-01'
end_date = '2018-01-01'
SRC_DATA_FILENAME = 'nicehash_ETHBTC_data_days.pk1'

try:
    nicehash_data = pd.read_pickle(SRC_DATA_FILENAME)
    print('File data found...reading ETHBTC data')
except FileNotFoundError:
    print('File not found...downloading the ETHBTC data')
    nicehash_data = data.DataReader('ETHBTC', 'yahoo', start_date, end_date)
    nicehash_data.to_pickle(SRC_DATA_FILENAME)

nicehash_data['Open-Close'] = nicehash_data.open-nicehash_data.close
nicehash_data['High-Low'] = nicehash_data.high-nicehash_data.low
nicehash_data = nicehash_data.dropna()
X = nicehash_data[['Open-Close', 'High-Low']]
Y = np.where(nicehash_data['close'].shift(-1) > nicehash_data['close'], 1, -1)

split_ratio = 0.8
split_value = int(split_ratio * len(nicehash_data))
X_train = X[:split_value]
Y_train = Y[:split_value]
X_test = X[split_value:]
Y_test = Y[split_value:]


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, Y_train)
accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
accuracy_test = accuracy_score(Y_test, knn.predict(X_test))


nicehash_data['Predicted_Signal'] = knn.predict(X)
nicehash_data['ETHBTC_Returns'] = np.log(nicehash_data['close'] /
                                         nicehash_data['close'].shift(1))


def calculate_return(df, split_value, symbol):
    cum_ETHBTC_return = df[split_value:]['%s_Returns' % symbol].cumsum() * 100
    df['Strategy_Returns'] = df['%s_Returns' %
                                symbol] * df['Predicted_Signal'].shift(1)
    return cum_ETHBTC_return


def calculate_strategy_return(df, split_value):
    cum_strategy_return = df[split_value:]['Strategy_Returns'].cumsum() * 100
    return cum_strategy_return


cum_ETHBTC_return = calculate_return(
    nicehash_data, split_value=len(X_train), symbol='ETHBTC')
cum_strategy_return = calculate_strategy_return(
    nicehash_data, split_value=len(X_train))


def plot_chart(cum_symbol_return, cum_strategy_return, symbol):
    plt.figure(figsize=(10, 5))
    plt.plot(cum_symbol_return, label='%s Returns' % symbol)
    plt.plot(cum_strategy_return, label='Strategy Returns')
    plt.legend()
    plt.grid()
    plt.show()


plot_chart(cum_ETHBTC_return, cum_strategy_return, symbol='ETHBTC')


# print(accuracy_train, accuracy_test)

# nicehash_data['Predicted_Signal']=knn.predict(X)
# nicehash_data['ETHBTC_Returns']=np.log(nicehash_data['close']/
#                                  nicehash_data['close'].shift(1))
# cum_ETHBTC_return=nicehash_data[split_value:]['ETHBTC_Returns'].cumsum()*100

# nicehash_data['Strategy_Returns']=nicehash_data['ETHBTC_Returns'] * nicehash_data['Predicted_Signal'].shift(1)
# cum_strategy_return=nicehash_data[split_value:]['Strategy_Returns'].cumsum()*100

# plt.figure(figsize=(10,5))
# plt.plot(cum_ETHBTC_return,label='ETHBTC Returns')
# plt.plot(cum_strategy_return,label='Strategy Returns')
# plt.legend()
# plt.show()


def sharpe_ratio(symbol_returns, strategy_returns):
    strategy_std = strategy_returns.std()
    sharpe = (strategy_returns-symbol_returns)/strategy_std
    return sharpe.mean()


print(sharpe_ratio(cum_strategy_return, cum_ETHBTC_return))
