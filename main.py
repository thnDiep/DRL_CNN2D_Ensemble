from DataLoader.DataLoader import YahooFinanceDataLoader
from DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from BaselineModels.VanillaInput.Train import Train as DeepRL
from utils import make_investment, save_action, load_action, get_data, label_data, ensemble_rule_based
from CNNModel import cnnpred_2d
from Evaluation import Evaluation
import pandas_ta as ta
import pandas as pd

import keras
from keras.models import load_model
import argparse
import os
import sklearn
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
import torch
import random
import numpy as np
from keras.callbacks import ReduceLROnPlateau

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='our',
                    help='[dqn, dqn_pi, dqn_ci, dqn_ti' # baseline models
                         'cnn2d_pi, cnn2d_ci, '
                         'random_forest, svm, rule_based'
                         'ours]')
parser.add_argument('-t', '--trader', type=str, default='test', help='[train, test, train_test]')
parser.add_argument('-d', '--dataset', default="AAL", help='Name of the data inside the Data folder')
parser.add_argument('-n', '--nep', type=int, default=1, help='Number of episodes')
parser.add_argument('-w', '--window_size', type=int, default=10, help='Window size for sequential models')
parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
args = parser.parse_args()

DATA_LOADERS = {
    'BTC-USD': YahooFinanceDataLoader('BTC-USD',
                                      split_point='2023-01-01',
                                      load_from_file=True),

    'GE': YahooFinanceDataLoader('GE',
                                 split_point='2023-01-01',
                                 load_from_file=True),

    'GOOGL': YahooFinanceDataLoader('GOOGL',
                                    split_point='2023-01-01',
                                    load_from_file=True),
}


class SensitivityRun:
    def __init__(self,
                 dataset_name,
                 gamma,
                 batch_size,
                 replay_memory_size,
                 feature_size,
                 target_update,
                 n_episodes,
                 n_step,
                 window_size,
                 device,
                 model_name,
                 transaction_cost=0):
        self.data_loader = DATA_LOADERS[dataset_name]
        self.dataset_name = dataset_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.feature_size = feature_size
        self.target_update = target_update
        self.n_episodes = n_episodes
        self.n_step = n_step
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.device = device
        self.model_name = model_name

        self.STATE_MODE_WINDOWED = 1    # window ohlc
        self.STATE_MODE_OUR = 2         # ours
        self.STATE_MODE_WINDOWED_PI = 3 # window ohlc + predictive indicators
        self.STATE_MODE_WINDOWED_CI = 4 # window ohlc + confirmatory indicators
        self.STATE_MODE_WINDOWED_TI = 5 # window ohlc + predictive + confirmatory indicators
        self.state_mode = None

        self.dataTrain = None
        self.dataTest = None

        self.x_train = None
        self.x_valid = None
        self.x_test = None

        self.y_train = None
        self.y_valid = None
        self.y_test = None

        self.dqn_agent = None
        self.ensemble_model = None
        self.cnn2d_pi_model = None
        self.cnn2d_ci_model = None

        self.early_stopping = None

        self.reset()

    def reset(self):
        self.load_data()
        self.load_agents()

    def load_data(self):
        if self.model_name == 'cnn2d_pi' or self.model_name == 'cnn2d_ci' \
            or self.model_name == 'random_forest' or self.model_name == 'svm' or self.model_name == 'rule_based': # to get the y_test
            x_train, y_train = get_data(self.data_loader.data_train, self.window_size, self.model_name)

            self.x_test, self.y_test = get_data(self.data_loader.data_test, self.window_size, self.model_name)

            self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_train,
                                                                                      y_train,
                                                                                      test_size=0.1,
                                                                                      shuffle=False)

        else:
            if self.model_name == 'dqn':
                self.state_mode = self.STATE_MODE_WINDOWED
            elif self.model_name == 'ours':
                self.state_mode = self.STATE_MODE_OUR
            elif self.model_name == 'dqn_pi':
                self.state_mode = self.STATE_MODE_WINDOWED_PI
            elif self.model_name == 'dqn_ci':
                self.state_mode = self.STATE_MODE_WINDOWED_CI
            elif self.model_name == 'dqn_ti':
                self.state_mode = self.STATE_MODE_WINDOWED_TI

            self.dataTrain = DataAutoPatternExtractionAgent(self.data_loader.data_train,
                                                            self.state_mode,
                                                            self.dataset_name,
                                                            'train',
                                                            'action_stock_data',
                                                            self.device,
                                                            self.gamma,
                                                            self.n_step,
                                                            self.batch_size,
                                                            self.window_size,
                                                            self.transaction_cost)

            self.dataTest = DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                                           self.state_mode,
                                                           self.dataset_name,
                                                           'test',
                                                           'action_stock_data',
                                                           self.device,
                                                           self.gamma,
                                                           self.n_step,
                                                           self.batch_size,
                                                           self.window_size,
                                                           self.transaction_cost)

    def load_agents(self):
        if self.model_name == 'cnn2d_pi':
            self.early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=100, verbose=0,
                mode='auto', baseline=None, restore_best_weights=False
            )

            self.cnn2d_pi_model = cnnpred_2d(self.window_size, 11, [8, 8, 8])
            self.cnn2d_pi_model.compile(optimizer='adam',
                                        loss='sparse_categorical_crossentropy',
                                        metrics=['accuracy'])

        elif self.model_name == 'cnn2d_ci':
            self.early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=100, verbose=0,
                mode='auto', baseline=None, restore_best_weights=False
            )

            self.cnn2d_ci_model = cnnpred_2d(self.window_size, 12, [8, 8, 8])
            self.cnn2d_ci_model.compile(optimizer='adam',
                                        loss='sparse_categorical_crossentropy',
                                        metrics=['accuracy'])

        elif self.model_name == 'dqn' \
                or self.model_name == 'dqn_pi' \
                or self.model_name == 'dqn_ci' \
                or self.model_name == 'dqn_ti' \
                or self.model_name == 'ours':
            self.dqn_agent = DeepRL(self.data_loader,
                                    self.dataTrain,
                                    self.dataTest,
                                    self.dataset_name,
                                    self.model_name,
                                    self.state_mode,
                                    self.window_size,
                                    self.transaction_cost,
                                    BATCH_SIZE=self.batch_size,
                                    GAMMA=self.gamma,
                                    ReplayMemorySize=self.replay_memory_size,
                                    TARGET_UPDATE=self.target_update,
                                    n_step=self.n_step)

    def train(self):
        if self.model_name == 'cnn2d_pi':
            self.cnn2d_pi_model.fit(self.x_train, self.y_train,
                                    epochs=200, batch_size=32,
                                    callbacks=[self.early_stopping],
                                    validation_data=(self.x_valid, self.y_valid),
                                    shuffle=False)

            self.cnn2d_pi_model.save(f'Models/{self.dataset_name}/{self.model_name}', save_format='tf')
            actions = np.concatenate((self.y_train, self.y_valid))
            save_action(actions, f'{self.dataset_name}/train/', self.model_name)

        elif self.model_name == 'cnn2d_ci':
            self.cnn2d_ci_model.fit(self.x_train, self.y_train,
                                    epochs=200, batch_size=32,
                                    callbacks=[self.early_stopping],
                                    validation_data=(self.x_valid, self.y_valid),
                                    shuffle=False)

            self.cnn2d_ci_model.save(f'Models/{self.dataset_name}/{self.model_name}', save_format='tf')
            actions = np.concatenate((self.y_train, self.y_valid))
            save_action(actions, f'{self.dataset_name}/train/', self.model_name)

        elif self.model_name == 'dqn' \
                or self.model_name == 'dqn_pi' \
                or self.model_name == 'dqn_ci' \
                or self.model_name == 'dqn_ti' \
                or self.model_name == 'ours':
            action_list = self.dqn_agent.train(self.n_episodes)
            save_action(action_list, f'{self.dataset_name}/train/', self.model_name)

        elif self.model_name == 'random_forest' \
                or self.model_name == 'svm':
            action_ohlc = load_action(f'{self.dataset_name}/train/dqn')
            action_leading = load_action(f'{self.dataset_name}/train/cnn2d_pi')
            action_lagging = load_action(f'{self.dataset_name}/train/cnn2d_ci')

            x_train = np.column_stack((action_ohlc, action_leading, action_lagging))
            y_train = np.concatenate((self.y_train, self.y_valid))

            if self.model_name == 'random_forest':
                self.ensemble_model = RandomForestClassifier(n_estimators=200, random_state=42)
                self.ensemble_model.fit(x_train, y_train)
            else:
                self.ensemble_model = svm.SVC(kernel='linear', C=1)
                self.ensemble_model.fit(x_train, y_train)

    def test(self):
        if self.model_name == 'cnn2d_pi':
            self.cnn2d_pi_model = load_model(f'Models/{dataset_name}/cnn2d_pi')
            y_pred = self.cnn2d_pi_model.predict(self.x_test)
            action_pred = np.argmax(y_pred, axis=1)
            save_action(action_pred, f'{self.dataset_name}/test/', self.model_name)

            # evaluate
            make_investment(self.data_loader.data_test, action_pred, 'action', self.window_size)
            ev_agent = Evaluation(self.data_loader.data_test, 'action', f'Results/{self.dataset_name}-{self.model_name}', 1000, 0)
            ev_agent.evaluate()

            print(classification_report(action_pred, self.y_test))

        elif self.model_name == 'cnn2d_ci':
            self.cnn2d_ci_model = load_model(f'Models/{dataset_name}/cnn2d_ci')
            y_pred = self.cnn2d_ci_model.predict(self.x_test)
            action_pred = np.argmax(y_pred, axis=1)
            save_action(action_pred, f'{self.dataset_name}/test/', self.model_name)

            # evaluate
            make_investment(self.data_loader.data_test, action_pred, 'action', self.window_size)
            ev_agent = Evaluation(self.data_loader.data_test, 'action', f'Results/{self.dataset_name}-{self.model_name}', 1000, 0)
            ev_agent.evaluate()

            print(classification_report(action_pred, self.y_test))

        elif self.model_name == 'dqn' \
                or self.model_name == 'dqn_pi' \
                or self.model_name == 'dqn_ci' \
                or self.model_name == 'dqn_ti'\
                or self.model_name == 'ours':
            ev_agent, action_list = self.dqn_agent.test()
            # ev_agent.plot_trading_process()
            save_action(action_list, f'{self.dataset_name}/test/', self.model_name)

        elif self.model_name == 'random_forest' \
                or self.model_name == 'svm'\
                or self.model_name == 'rule_based':
            action_ohlc = load_action(f'{self.dataset_name}/test/dqn')
            action_leading = load_action(f'{self.dataset_name}/test/cnn2d_pi')
            action_lagging = load_action(f'{self.dataset_name}/test/cnn2d_ci')
            action_matrix = np.column_stack((action_ohlc, action_leading, action_lagging))

            if self.model_name == 'rule_based':
                y_pred = ensemble_rule_based(action_ohlc, action_leading, action_lagging)
            else:
                y_pred = self.ensemble_model.predict(action_matrix)

            # evaluate
            make_investment(self.data_loader.data_test, y_pred, 'action', self.window_size)
            ev_agent = Evaluation(self.data_loader.data_test, 'action', f'Results/{self.dataset_name}-{self.model_name}', 1000, 0)
            ev_agent.evaluate()

            print(classification_report(y_pred, self.y_test))

if __name__ == '__main__':
    set_random_seed(42)
    n_step = 8
    window_size = args.window_size
    dataset_name = args.dataset
    n_episodes = args.nep
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("using: ", device)
    feature_size = 64
    target_update = 5

    gamma = 0.9
    batch_size = 16
    replay_memory_size = 32

    trader = args.trader
    model_name = args.model

    print(pd.__version__)
    print(ta.__version__)
    print(np.__version__)
    print(sklearn.__version__)



    run = SensitivityRun(
        dataset_name,
        gamma,
        batch_size,
        replay_memory_size,
        feature_size,
        target_update,
        n_episodes,
        n_step,
        window_size,
        device,
        model_name,
        transaction_cost=0)
    run.reset()

    if trader == 'train' or trader == 'train_test':
        run.train()

    if trader == 'test' or trader == 'train_test':
        run.test()
