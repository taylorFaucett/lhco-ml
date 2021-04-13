# Silnence output of tensorflow/keras about GPU status
import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Keras/TF imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


kernel_constraint = max_norm(3)
bias_constraint = max_norm(3)


def get_CNN(model, tp, input_shape):
    padding = "same"
    activation = "relu"
    model.add(
        Conv2D(32, (3, 3), padding=padding, activation=activation, input_shape=input_shape,)
    )
    model.add(
        Conv2D(
            int(tp["filter_units_1"]),
            kernel_size=int(tp["kernel_size_1"]),
            padding=padding,
            activation=activation,
            kernel_constraint=kernel_constraint,
            bias_constraint=kernel_constraint,
        )
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            int(tp["filter_units_2"]),
            kernel_size=int(tp["kernel_size_2"]),
            padding=padding,
            activation=activation,
            kernel_constraint=kernel_constraint,
            bias_constraint=kernel_constraint,
        )
    )
    model.add(MaxPooling2D(pool_size=int(tp["max_pool_1"]), padding="same"))
    model.add(
        Conv2D(
            int(tp["filter_units_3"]),
            kernel_size=int(tp["kernel_size_3"]),
            padding=padding,
            activation=activation,
            kernel_constraint=kernel_constraint,
            bias_constraint=kernel_constraint,
        )
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            int(tp["filter_units_4"]),
            kernel_size=int(tp["kernel_size_3"]),
            padding=padding,
            activation=activation,
            kernel_constraint=kernel_constraint,
            bias_constraint=kernel_constraint,
        )
    )    
    model.add(MaxPooling2D(pool_size=int(tp["max_pool_2"]), padding=padding))
    model.add(Flatten())
    model.add(
            Dense(int(tp["dense_units_1"]), 
            activation=activation, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=kernel_constraint)
            )
    model.add(Dropout(tp["dropout_1"]))
    model.add(
            Dense(int(tp["dense_units_1"]), 
            activation=activation, 
            kernel_constraint=kernel_constraint, 
            bias_constraint=kernel_constraint)
            )
    model.add(Dropout(tp["dropout_2"]))
    return model

def get_CNN_slim(model, tp, input_shape):
    padding = "same"
    activation = "relu"
    kernel_size = 3
    pool_size = 2
    model.add(
        Conv2D(
            256,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            kernel_constraint=kernel_constraint,
            bias_constraint=kernel_constraint,
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, padding=padding))
    model.add(
        Conv2D(
            256,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            kernel_constraint=kernel_constraint,
            bias_constraint=kernel_constraint,
        )
    )
    model.add(Flatten())

    model.add(Flatten())
    model.add(Dense(300, activation=activation, kernel_constraint=kernel_constraint, bias_constraint=kernel_constraint,))
    model.add(Dropout(0.1))
    model.add(Dense(300, activation=activation, kernel_constraint=kernel_constraint, bias_constraint=kernel_constraint,))
    model.add(Dropout(0.1))
    return model


def get_HL(model, tp, input_shape):
    model.add(Flatten(input_shape=(input_shape,)))
    for dense_ix in range(int(tp["dense_layers"])):
        model.add(
            Dense(
                tp["dense_units"],
                activation="relu",
                kernel_initializer=kernel_initializer,
                kernel_constraint=kernel_constraint,
                bias_constraint=kernel_constraint,
            )
        )
        if dense_ix - 1 < int(tp["dense_layers"]):
            model.add(Dropout(tp["dropout"]))
    return model


def get_model(run_type, tp, input_shape):
    model = Sequential()
    if run_type == "CNN_1" or "CNN_2":
        model = get_CNN(model, tp, input_shape)
    elif run_type == "HL":
        model = get_HL(model, tp, input_shape)

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=tp["learning_rate"]),
        metrics=["accuracy", metrics.AUC(name="auc")],
    )
    
    return model
