import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ref: https://github.com/maleakhiw/stock-prediction
# author: Hoseinzade and Haratizadeh 2019
def cnnpred_2d(sequence_length, n_feature, n_filters, dropout_rate=0.001):
    """
    Build model using architecture that is specified on the paper
    (Hoseinzade and Haratizadeh).
    """

    model = keras.Sequential([
        # Layer 1
        keras.Input(shape=(sequence_length, n_feature, 1)),
        layers.Conv2D(n_filters[0], (1, n_feature), activation="relu"),

        # Layer 2
        layers.Conv2D(n_filters[1], (3, 1), activation="relu"),
        layers.MaxPool2D(pool_size=(2, 1)),

        # Layer 3
        layers.Conv2D(n_filters[2], (3, 1), activation="relu"),
        layers.MaxPool2D(pool_size=(2, 1)),

        # FFNN
        layers.Flatten(),
        layers.Dense(3, activation='softmax')
    ])

    return model