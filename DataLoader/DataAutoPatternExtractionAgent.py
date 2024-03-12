from .Data import Data
import numpy as np
from utils import load_action
from sklearn.preprocessing import MinMaxScaler

#ref: https://github.com/MehranTaghian/DQN-Trading
#author: Taghian, Mehran and Asadi, Ahmad and Safabakhsh, Reza

class DataAutoPatternExtractionAgent(Data):
    def __init__(self, data, state_mode, dataset_name, data_type, action_name, device, gamma,
                 n_step=4, batch_size=50, window_size=1, transaction_cost=0.0):

        start_index_reward = 0 if state_mode != 1 else window_size - 1
        super().__init__(data, action_name, device, gamma, n_step, batch_size,
                         start_index_reward=start_index_reward,
                         transaction_cost=transaction_cost)
        self.data_type = data_type
        self.data_kind = 'stock_data'
        self.state_mode = state_mode
        self.dataset_name = dataset_name

        if state_mode == 1: # OHLC windowed
            self.state_size = window_size * 4
            np_array = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

            for i in range(window_size, np_array.shape[0]):
                temp_states = np_array[i - window_size: i]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        elif state_mode == 2: # data for our model
            self.state_size = 6 * window_size
            action_leading = load_action(f'{self.dataset_name}/{self.data_type}/cnn2d_pi')
            action_lagging = load_action(f'{self.dataset_name}/{self.data_type}/cnn2d_ci')

            np_array = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
            actions = np.vstack((np.ones(np_array.shape[0]), np.ones(np_array.shape[0]))).T
            np_array = np.hstack((np_array, actions))

            for i in range(window_size, np_array.shape[0]):
                temp_states = np_array[i - window_size: i]
                temp_states[-1][4] = action_leading[i - window_size]
                temp_states[-1][5] = action_lagging[i - window_size]

                temp_states = temp_states.reshape(-1)

                self.states.append(temp_states)

        elif state_mode == 3:  # ohlc + predictive indicators
            self.state_size = window_size * (11 + 4)

            ohlc = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

            leading = data.loc[:, ['volume', 'rsi', 'stochk', 'stochd', 'cci',
                                   'willr', 'mom', 'roc', 'trix', 'trixs', 'cmo']].values
            min_max_scaler = MinMaxScaler()
            leading = min_max_scaler.fit_transform(leading)

            np_array = np.concatenate((ohlc, leading), axis=1)

            for i in range(window_size, np_array.shape[0]):
                temp_states = np_array[i - window_size: i]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        elif state_mode == 4:  # ohlc + confirmatory indicators
            self.state_size = window_size * (12 + 4)

            ohlc = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

            lagging = data.loc[:, ['sma_50', 'sma_200',
                                   'ema_10', 'ema_20', 'ema_50', 'ema_200',
                                   'macd', 'macdh', 'macds',
                                   'tema', 'kama', 'wma']].values
            min_max_scaler = MinMaxScaler()
            lagging = min_max_scaler.fit_transform(lagging)

            np_array = np.concatenate((ohlc, lagging), axis=1)

            for i in range(window_size, np_array.shape[0]):
                temp_states = np_array[i - window_size: i]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)

        elif state_mode == 5:  # ohlc + predictive + confirmatory indicators
            self.state_size = window_size * (23 + 4)

            ohlc = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

            indicator = data.loc[:, ['volume', 'rsi', 'stochk', 'stochd', 'cci',
                                     'willr', 'mom', 'roc', 'trix', 'trixs', 'cmo',
                                     'sma_50', 'sma_200',
                                     'ema_10', 'ema_20', 'ema_50', 'ema_200',
                                     'macd', 'macdh', 'macds',
                                     'tema', 'kama', 'wma']].values
            min_max_scaler = MinMaxScaler()
            indicator = min_max_scaler.fit_transform(indicator)

            np_array = np.concatenate((ohlc, indicator), axis=1)

            for i in range(window_size, np_array.shape[0]):
                temp_states = np_array[i - window_size: i]
                temp_states = temp_states.reshape(-1)
                self.states.append(temp_states)