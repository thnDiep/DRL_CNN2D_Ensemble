import warnings
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from utils import add_leading_lagging_indicators

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import ast
from pathlib import Path

#ref: https://github.com/MehranTaghian/DQN-Trading
#author: Taghian, Mehran and Asadi, Ahmad and Safabakhsh, Reza

class YahooFinanceDataLoader:
    def __init__(self, dataset_name, split_point, begin_date=None, end_date=None, load_from_file=False):
        warnings.filterwarnings('ignore')
        self.DATA_NAME = dataset_name
        self.DATA_PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent, f'Data/{self.DATA_NAME}/')

        self.DATA_FILE = dataset_name + '.csv'

        self.split_point = split_point
        self.begin_date = begin_date
        self.end_date = end_date

        if not load_from_file:
            self.data = self.load_data()
            self.normalize_data()
            self.data.to_csv(f'{self.DATA_PATH}{self.DATA_NAME}_processed.csv', index=True)
        else:
            self.data = pd.read_csv(f'{self.DATA_PATH}{self.DATA_NAME}_processed.csv')
            self.data.set_index('Date', inplace=True)
            self.normalize_data()

        if begin_date is not None:
            self.data = self.data[self.data.index >= begin_date]

        if end_date is not None:
            self.data = self.data[self.data.index <= end_date]

        if type(split_point) == str:
            self.data_train = self.data[self.data.index < split_point]
            self.data_test = self.data[self.data.index >= split_point]
        elif type(split_point) == int:
            self.data_train = self.data[:split_point]
            self.data_test = self.data[split_point:]
        else:
            raise ValueError('Split point should be either int or date!')

        self.data_train_with_date = self.data_train.copy()
        self.data_test_with_date = self.data_test.copy()
        # self.plot_dataset()

        self.data_train.reset_index(drop=True, inplace=True)
        self.data_test.reset_index(drop=True, inplace=True)

    def load_data(self):
        data = pd.read_csv(f'{self.DATA_PATH}{self.DATA_FILE}')
        data.dropna(inplace=True)
        data.set_index('Date', inplace=True)
        data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
        add_leading_lagging_indicators(data)
        data = data.drop(['Adj Close'], axis=1)
        data['action'] = "None"
        return data

    def normalize_data(self):
        min_max_scaler = MinMaxScaler()
        self.data['open_norm'] = min_max_scaler.fit_transform(self.data.open.values.reshape(-1, 1))
        self.data['high_norm'] = min_max_scaler.fit_transform(self.data.high.values.reshape(-1, 1))
        self.data['low_norm'] = min_max_scaler.fit_transform(self.data.low.values.reshape(-1, 1))
        self.data['close_norm'] = min_max_scaler.fit_transform(self.data.close.values.reshape(-1, 1))

    def plot_dataset(self):
        plt.figure(figsize=(9, 5))
        df1 = pd.Series(self.data_train_with_date.close, index=self.data.index)
        df2 = pd.Series(self.data_test_with_date.close, index=self.data.index)
        ax = df1.plot(color='b', label='Train', linewidth=1)
        df2.plot(ax=ax, color='r', label='Test', linewidth=1)
        plt.legend(loc='upper left')
        ax.set(xlabel='Time', ylabel='Close Price')
        step = max(len(df1) // 4, 1)
        plt.xticks(range(0, len(df1), step), df1.index[::step])
        plt.savefig(f'{self.DATA_PATH}{self.DATA_NAME}_image.jpg')
