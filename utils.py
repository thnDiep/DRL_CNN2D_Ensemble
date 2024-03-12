import pandas_ta as ta
from PIL import Image
import os
import numpy as np
from Action import Action
from pathlib import Path
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler

def add_leading_lagging_indicators(data):
    CustomStrategy = ta.Strategy(
        name="Used Indicators",
        description="Predictive and confirmatory indicators",
        ta=[
            # PREDICTIVE INDICATOR
            {"kind": "rsi", "length": 14, "col_names": ("rsi")},
            {"kind": "stoch", "col_names": ("stochk", "stochd")},
            {"kind": "cci", "col_names": ("cci")},
            {"kind": "willr", "col_names": ("willr")},
            {"kind": "mom", "length": 10, "col_names": ("mom")},
            {"kind": "roc", "col_names": ("roc")},
            {"kind": "trix", "length": 5, "col_names": ("trix", "trixs")},
            {"kind": "cmo", "length": 5, "col_names": ("cmo")},

            # CONFIRMATORY INDICATOR
            {"kind": "sma", "length": 50, "col_names": ("sma_50")},
            {"kind": "sma", "length": 200, "col_names": ("sma_200")},
            {"kind": "ema", "length": 10, "col_names": ("ema_10")},
            {"kind": "ema", "length": 20, "col_names": ("ema_20")},
            {"kind": "ema", "length": 50, "col_names": ("ema_50")},
            {"kind": "ema", "length": 200, "col_names": ("ema_200")},
            {"kind": "macd", "fast": 8, "slow": 21, "col_names": ("macd", "macdh", "macds")},
            {"kind": "tema", "length": 5, "col_names": ("tema")},
            {"kind": "wma", "length": 5, "col_names": ("wma")},
            {"kind": "kama", "length": 5, "col_names": ("kama")},
        ]
    )

    data.ta.strategy(CustomStrategy)
    data.dropna(inplace=True)


def get_data(df, window_size, model_name):
    if model_name == 'cnn2d_pi':
        data = df.loc[:, ['volume', 'rsi', 'stochk', 'stochd', 'cci',
                          'willr', 'mom', 'roc', 'trix', 'trixs', 'cmo']].values
    elif model_name == 'cnn2d_ci':
        data = df.loc[:, ['sma_50', 'sma_200',
                          'ema_10', 'ema_20', 'ema_50', 'ema_200',
                          'macd', 'macdh', 'macds',
                          'tema', 'kama', 'wma']].values
    else:
        data = df.loc[:, ['close']].values

    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    x = []
    for i in range(window_size, data.shape[0]):
        x.append(data[i-window_size:i])

    x = np.expand_dims(x, axis=-1)

    labels = label_data(df, window_size)
    y = np.array(labels)
    return x, y

def make_investment(data, action_list, action_name, window_size):
    code_to_action = {0: 'buy', 1: 'None', 2: 'sell'}
    data[action_name] = 'None'
    i = window_size
    for a in action_list:
        data[action_name][i] = code_to_action[a]
        i += 1


def save_action(action_list, path, file_name):
    directory = f'Results/actions/{path}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, file_name)
    array_str = ",".join(map(str, action_list))

    with open(file_path, "w") as file:
        file.write(array_str)


def load_action(file_name):
    directory = f'Results/actions/'
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'r') as file:
        data_str = file.read().strip().split(',')
    data_array = []
    for value in data_str:
        try:
            data_array.append(int(value))
        except ValueError:
            pass

    data_array = np.array(data_array)
    return data_array



def label_data(data, window_size):
    label_threshold = 0.005
    data = data[window_size:]
    ensemble_y_true = np.ones(len(data), dtype=int)
    last_action = 1

    for i in range(len(data) - 1):
        current_price = data.iloc[i]['close']
        next_price = data.iloc[i + 1]['close']

        changes = (next_price - current_price) / current_price

        if changes >= label_threshold or (last_action == 0 and 0 <= changes < label_threshold):
            last_action = 0 # BUY
            ensemble_y_true[i] = 0
        elif changes < -label_threshold or (last_action == 2 and 0 > changes >= -label_threshold):
            last_action = 2 # SELL
            ensemble_y_true[i] = 2
        else:
            last_action = 1 # HOLD
            ensemble_y_true[i] = 1
    return np.array(ensemble_y_true)

def ensemble_rule_based(actions1, actions2, actions3):
    actions = np.array([actions1, actions2, actions3]).T
    final_actions = []

    for row in actions:
        buy_signal_count = 0
        sell_signal_count = 0
        hold_signal_count = 0

        for action in row:
            if action == Action.BUY.value:
                buy_signal_count += 1
            elif action == Action.SELL.value:
                sell_signal_count += 1
            else:
                hold_signal_count += 1

        final_action = Action.NONE.value
        if buy_signal_count > sell_signal_count and buy_signal_count > hold_signal_count:
            final_action = Action.BUY.value
        elif sell_signal_count > buy_signal_count and sell_signal_count > hold_signal_count:
            final_action = Action.SELL.value

        final_actions.append(final_action)

    return final_actions
