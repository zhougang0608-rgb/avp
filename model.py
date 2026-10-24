"""Baseline classifier with parameters for controlled Version 4 comparisons."""
from tensorflow import keras
from tensorflow.keras import layers

from settings import NAMES


def build_model(input_shape, filters=8, blocks=1, dropout=0.0):
    model = keras.Sequential([layers.Input(shape=input_shape)])
    for _ in range(blocks):
        model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation="relu"))
    if dropout:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(len(NAMES), activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
