import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf


def dice_loss(y_true, y_pred):
    def dice_coeff():
        smooth = 1
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_mean(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_mean(y_true_f) + tf.reduce_mean(y_pred_f) + smooth)
        return score

    return 1 - dice_coeff()


def bce_dice_loss(y_true, y_pred):
    losses = keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return losses


