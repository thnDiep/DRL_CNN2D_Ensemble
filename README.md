# Enhancing Financial Market Prediction with Reinforcement Learning and Ensemble Learning
## Abstract 
In this paper, we introduce an innovative approach to predicting stock market trends, effectively merging Convolutional Neural Networks (CNN) with Reinforcement Learning (RL). This method focuses on transforming data from both lagging and leading indicators, which reflect past market performance and predict future trends and provide trading signals, respectively, into image data for CNN processing. A key highlight of our approach is the use of imaging technology in the representation and analysis of financial data, offering a novel insight into this traditional field with cutting-edge technology. Employing images to represent financial data provides a unique and in-depth view of the market, enabling our model to detect and learn from complex patterns not easily identifiable through standard analytical methods. In addition to integrating important indicators, we also consider stock price time series to develop a multifaceted feature system, providing a comprehensive understanding of market conditions. Our experimental results reveal that our method not only surpasses traditional forecasting models in accuracy but also opens up a practical approach to the integration of multi-faceted financial trend analysis, thereby enhancing our understanding and accuracy in stock market predictions.

## Structure description
* main.py: the entry point of the application.
* ReplayMemory.py: to manage the replay memory.
* utils.py: support functions.
* Evaluation.py: Apply the Stop Loss and Take Profit method to evaluate the model
* CNNModel.py: structure of 2D-CNN, base on [Maleakhi Wijaya](https://github.com/maleakhiw/stock-prediction)
* Data folder: contains 3 datasets GOOGL, GE and BTC-USD.
* DataLoader folder: perform data retrieval and data preprocessing.
* BaselineModels folder: contains a baseline model to compare with the proposed model, based on [DQN-Trading](https://github.com/MehranTaghian/DQN-Trading)
* Models folder: where to save models trained
* Results folder: contains train results and test results file.


## Requirements
* python: 3.11.5
* torch: 2.1.0
* pandas: 2.0.3
* pandas-ta: 0.3.14b0
* numpy: 1.24.3
* scikit-learn: 1.3.0
* keras: 2.14.0
* tensorflow: 2.14.0

## Usage
To run code, use command:
```shell
python main.py -t <trader> -m <model> -w <windowed-size> -d <dataset> -n <number-of-episodes>`
```
* `trader`
  * train: to train the model 
  * test: to test the model
  * train_test: model will be trained and tested
* `model`
  * dqn
  * dqn_pi
  * dqn_ci
  * dqn_ti
  * cnn2d_pi
  * cnn2d_ci
  * random_forest
  * svm
  * rule_based
  * ours
* `windowed-size`: default is 10
* `dataset`: GOOGL, GE, BTC-USD
* `number-of-episodes`: default is 1

Example, to test the proposed model in BTC-USD dataset:
```shell
python main.py -m ours -w 10 -d BTC-USD
```
