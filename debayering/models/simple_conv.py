import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, 
    Convolution2D,
)


def get_model(input_shape=(None, None, 4)): 
    inputs = Input(shape=input_shape)

    conv1 = Convolution2D(16, (3, 3), activation='tanh', padding='same')(inputs)
    conv2 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Convolution2D(12, (3, 3), activation='relu', padding='same')(conv2)
    x_out = tf.nn.depth_to_space(conv3, 2)
    return Model(inputs=inputs, outputs=x_out)


def get_model_flat(input_shape=(None, None)): 
    inputs = Input(shape=input_shape)
    inputs = tf.expand_dims(inputs, axis=-1)

    conv1 = Convolution2D(8, (5, 5), activation='tanh', padding='same')(inputs)
    conv2 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Convolution2D(12, (3, 3), activation='relu', padding='same')(conv2)
    x_out = Convolution2D(3, (3, 3), activation='relu', padding='same')(conv3)
    return Model(inputs=inputs, outputs=x_out)