import numpy as np
import tensorflow as tf

def categorical_crossentropy(y_true,y_pred):
    return tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true,y_pred,from_logits=True))