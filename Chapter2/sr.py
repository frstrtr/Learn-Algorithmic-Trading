import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pandas_datareader import data

start_date = '2014-01-01'
end_date = '2018-01-01'
SRC_DATA_FILENAME = 'nicehash_ETHBTC_data_days.pk1'
# SRC_DATA_FILENAME = 'nicehash_ETHBTC_data_minutes.pk1'

try:
    nicehash_data2 = pd.read_pickle(SRC_DATA_FILENAME)
except FileNotFoundError:
    nicehash_data2 = data.DataReader('GOOG', 'yahoo', start_date, end_date)
    nicehash_data2.to_pickle(SRC_DATA_FILENAME)

print(nicehash_data2)
nicehash_data = nicehash_data2  # .tail(620)
lows = nicehash_data['low']
highs = nicehash_data['high']

nicehash_data_signal = pd.DataFrame(index=nicehash_data.index)
nicehash_data_signal['price'] = nicehash_data['close']


# bin_width - the time window in the price that is used to calculate the resistance and support levels.
def trading_support_resistance(data, bin_width=12, margin=38, tolerance = 2):
    margin = margin / 100  # convert %
    data['sup_tolerance'] = pd.Series(np.zeros(len(data)))
    data['res_tolerance'] = pd.Series(np.zeros(len(data)))
    data['sup_count'] = pd.Series(np.zeros(len(data)))
    data['res_count'] = pd.Series(np.zeros(len(data)))
    data['sup'] = pd.Series(np.zeros(len(data)))
    data['res'] = pd.Series(np.zeros(len(data)))
    data['positions'] = pd.Series(np.zeros(len(data)))
    data['signal'] = pd.Series(np.zeros(len(data)))
    in_support = 0
    in_resistance = 0

    for x in range((bin_width - 1) + bin_width, len(data)):
        data_section = data[x - bin_width:x + 1]
        support_level = min(data_section['price'])
        resistance_level = max(data_section['price'])
        range_level = resistance_level-support_level
        data['res'][x] = resistance_level
        data['sup'][x] = support_level

        # The level of support and resistance is calculated by taking the maximum and minimum price and then subtracting and adding a __% margin.
        data['sup_tolerance'][x] = support_level + margin * range_level
        data['res_tolerance'][x] = resistance_level - margin * range_level

        if data['price'][x] >= data['res_tolerance'][x] and data['price'][x] <= data['res'][x]:
            in_resistance += 1
            data['res_count'][x] = in_resistance
        elif data['price'][x] <= data['sup_tolerance'][x] and data['price'][x] >= data['sup'][x]:
            in_support += 1
            data['sup_count'][x] = in_support
        else:
            in_support = 0
            in_resistance = 0
            # a buy order is sent when a price stays in the resistance
            # tolerance margin for 2 consecutive days, and that a sell order is sent when a price stays in
            # the support tolerance margin for 2 consecutive days.
        if in_resistance > 2:
            data['signal'][x] = 1
        elif in_support > 2:
            data['signal'][x] = 0
        else:
            data['signal'][x] = data['signal'][x-1]
    data['positions'] = data['signal'].diff()


trading_support_resistance(nicehash_data_signal)

'''

The Support and Resistance (SR) indicator
Momentum trading strategy based on support and resistance levels.

'''

# ETHBTC data window 200 days support and resistance
# We draw the highs and lows of the ETHBTC price.
# The green line represents the resistance level, and the red line represents the support level.
# To build these lines, we use the maximum value of the ETHBTC price and the minimum value of the ETHBTC price stored daily.
# After the 200th day (dotted vertical blue line), we will buy when we reach the support line, and sell when we reach the resistance line. In this example, we used
# 200 days so that we have sufficient data points to get an estimate of the trend.
# It is observed that the ETHBTC price will reach the resistance line around __date__. This means that we have a signal to enter a short position (sell).
# Once traded, we will wait to get out of this short position when the ETHBTC price will reach the support line.
# With this historical data, it is easily noticeable that this condition will not happen.
# This will result in carrying a short position in a rising market without having any signal to sell it, thereby resulting in a huge loss.

# basic support and resistance lines for the whole period
# After the gap __th day (dotted vertical blue line), we will buy when we reach the
gap = 50
# support line, and sell when we reach the resistance line. In this example, we used
# __ gap days so that we have sufficient data points to get an estimate of the trend.

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='ETH price in BTC')
highs.plot(ax=ax1, color='c', lw=2.)
lows.plot(ax=ax1, color='y', lw=2.)
plt.hlines(highs.head(gap).max(),
           lows.index.values[0], lows.index.values[-1], linewidth=2, color='g')
plt.hlines(lows.head(gap).min(),
           lows.index.values[0], lows.index.values[-1], linewidth=2, color='r')
plt.axvline(linewidth=2, color='b', x=lows.index.values[gap], linestyle=':')
# plt.show()

# __-day rolling window calculating resistance and support:
# buy order is sent when a price stays in the resistance
# tolerance margin for 2 consecutive days,
# and that a sell order is sent when a price stays in
# the support tolerance margin for 2 consecutive days.
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='ETH price in BTC')
nicehash_data_signal['sup'].plot(ax=ax1, color='g', lw=1.)
nicehash_data_signal['res'].plot(ax=ax1, color='b', lw=1.)
nicehash_data_signal['price'].plot(ax=ax1, color='r', lw=1.)
ax1.plot(nicehash_data_signal.loc[nicehash_data_signal.positions == 1.0].index,
         nicehash_data_signal         .price[nicehash_data_signal         .positions == 1.0], '^', markersize=7, color='m', label='buy')
ax1.plot(nicehash_data_signal.loc[nicehash_data_signal.positions == -1.0].index,
         nicehash_data_signal         .price[nicehash_data_signal         .positions == -1.0], 'v', markersize=7, color='k', label='sell')
plt.legend()
# plt.show()


# backtesting
# Set the initial capital
initial_capital = float(0.25)  # in BTC

positions = pd.DataFrame(index=nicehash_data_signal.index).fillna(0.0)
portfolio = pd.DataFrame(index=nicehash_data_signal.index).fillna(0.0)


positions['ETHBTC'] = nicehash_data_signal['signal']
portfolio['positions'] = (positions.multiply(
    nicehash_data_signal['price'], axis=0))
portfolio['cash'] = initial_capital - \
    (positions.diff().multiply(nicehash_data_signal['price'], axis=0)).cumsum()
portfolio['total'] = portfolio['positions'] + portfolio['cash']
portfolio.plot()
# plt.show()

# taking my last balance for review
print('My starting balance::', initial_capital, ' BTC')
print('My final balance::', portfolio['total'].iloc[-1], ' BTC')
print('My earnings (profit + loss -) ::', "%08.8f" %
      (portfolio['total'].iloc[-1] - initial_capital), '\tBTC')
print('My earnings (profit + loss -) ::', "%08.8f" %
      ((portfolio['total'].iloc[-1] - initial_capital)*651675.89), ' RUR')

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Portfolio value in BTC', xlabel='Days')
portfolio['total'].plot(ax=ax1, lw=2.)
ax1.plot(portfolio.loc[nicehash_data_signal.positions == 1.0].index,
         portfolio.total[nicehash_data_signal.positions == 1.0], '^', markersize=10, color='m')
ax1.plot(portfolio.loc[nicehash_data_signal.positions == -1.0].index,
         portfolio.total[nicehash_data_signal.positions == -1.0], 'v', markersize=10, color='k')
plt.show()


# todo iterate all possible combinations of margin x window
margin_max = 100 # max 100%
window_max = 100 # no more than half of the data set
marg, window = (0, 0)
initial_capital = float(0.25)  # in BTC
abs_max = []

# create empty list with zeroes margin_max x window_max
sol = [[0] * window_max for _ in range(margin_max)]

print()
print('Model iteration...')
print ('margin (%)\t window(days)\t balance (BTC)')

for marg in range(1, margin_max+1):  # margin 1..99 %

    for window in range(1, window_max+1):  # window length in days
        
        for tol in range(1,window-1): # tolerance iter for 1 to window-1 value
            # print(marg,'\t\t',window, end='\r')
            trading_support_resistance(nicehash_data_signal, window, marg, tol)

            positions = pd.DataFrame(index=nicehash_data_signal.index).fillna(0.0)
            portfolio = pd.DataFrame(index=nicehash_data_signal.index).fillna(0.0)

            positions['ETHBTC'] = nicehash_data_signal['signal']
            portfolio['positions'] = (positions.multiply(nicehash_data_signal['price'], axis=0))
            portfolio['cash'] = initial_capital - (positions.diff().multiply(nicehash_data_signal['price'], axis=0)).cumsum()
            portfolio['total'] = portfolio['positions'] + portfolio['cash']

            balance = portfolio['total'].iloc[-1] - initial_capital

            if balance > sol[marg-1][window-1]:
                sol[marg-1][window-1] = balance
            

    # print(marg, sol[marg-1])
    # print()
    max_earning = max(sol[marg-1])
    if max_earning:  # exclude 0
        print(marg, '\t\t', sol[marg-1].index(max_earning), '\t\t', "%08.8f" % max_earning)
        abs_max.append([marg, sol[marg-1].index(max_earning), max_earning])

print()
print('Absolute max profit:')
abs_max_transposed = list(map(list, zip(*abs_max)))

print("%08.8f" % max(abs_max_transposed[2]))
index_max = abs_max_transposed[2].index(max(abs_max_transposed[2]))
print('Margin =', abs_max_transposed[0][index_max],
      'Window =', abs_max_transposed[1][index_max])
