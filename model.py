"This is an implementation of Inception V1 module using tensorflow"
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def Conv1D_block(input, output_size, kernel, strides = 1, padding = "valid"):
    # 1x1 convo block
    output = tf.keras.layers.Conv1D(filter=output_size, kernel_size=kernel, strides=strides, padding=padding, activation="relu", kernel_initializer="he_normal")(input)
    return output

def Inception_block(input, filter_b1, filter_b2, filter_b3, filter_b4):
    "Input: filter_b1 = number of filters of first branch - shape = (1,)
    "       filter_b2 = number of filters of second branch - shape = (a, b) ------ a is for the first sublayer of this branch, b is for the second sublayer"
    "       filter_b3 = number of filters of third branch - shape = (a, b) ------ a is for the first sublayer of this branch, b is for the second sublayer"
    "       filter_b4 = number of filters of fourth branch - shape = (a, b) ------ a is for the first sublayer of this branch, b is for the second sublayer"

    #Branch 1x1
    b1 = Conv1D_block(input=input, output_size=filter_b1, kernel=1, strides=1)
    #Branch 1x1 and 3x3
    b2 = Conv1D_block(input=input, output_size=filter_b2[0], kernel=1, strides=1)
    b2 = Conv1D_block(input=b2, output_size=filter_b2[1], kernel=3, strides=1, padding="same")
    #Branch 1x1 and 5x5
    b3 = Conv1D_block(input=input, output_size=filter_b3[0], kernel=1, strides=1)
    b3 = Conv1D_block(input=b3, output_size=filter_b3[1], kernel=5, strides=1, padding="same")
    #Branch Max_pooling and 1x1
    b4 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(input)
    b4 = Conv1D_block(input=b4, output_size=filter_b4, kernel=1)

    output = tf.keras.layers.concatenate([b1, b2, b3, b4], axis = -1)

    return output

class GoogleNet:
    def __init__(self)

model.summary()