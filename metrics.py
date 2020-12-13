import numpy as np
import tensorflow as tf


def supervised_acc(y_true,y_pred):
    return tf.math.reduce_mean(tf.keras.metrics.categorical_accuracy(y_true,y_pred)).numpy()




    