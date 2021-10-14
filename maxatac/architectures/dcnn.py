import logging
from scipy import stats
from maxatac.utilities.system_tools import Mute


with Mute():
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.layers import (
        Input,
        Conv1D,
        MaxPooling1D,
        Lambda,
        BatchNormalization,
        Dense,
        Flatten
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    from maxatac.utilities.constants import KERNEL_INITIALIZER, INPUT_LENGTH, INPUT_CHANNELS, INPUT_FILTERS, \
        INPUT_KERNEL_SIZE, INPUT_ACTIVATION, OUTPUT_FILTERS, OUTPUT_KERNEL_SIZE, FILTERS_SCALING_FACTOR, DILATION_RATE, \
        OUTPUT_LENGTH, CONV_BLOCKS, PADDING, POOL_SIZE, ADAM_BETA_1, ADAM_BETA_2, DEFAULT_ADAM_LEARNING_RATE, \
        DEFAULT_ADAM_DECAY


def loss_function(
        y_true,
        y_pred,
        y_pred_min=0.0000001,  # 1e-7
        y_pred_max=0.9999999,  # 1 - 1e-7
        y_true_min=-0.5
):
    y_true = K.flatten(y_true)
    y_pred = tf.clip_by_value(
        K.flatten(y_pred),
        y_pred_min,
        y_pred_max
    )
    losses = tf.boolean_mask(
        tensor=-y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred),
        mask=K.greater_equal(y_true, y_true_min)
    )
    return tf.reduce_mean(input_tensor=losses)

def pearson(y_true, y_pred):
    import scipy.stats as measures
    import numpy as np
    x = y_true
    y = y_pred
    
    mx= K.cast(K.mean(x), dtype=np.float32)
    my= K.cast(K.mean(y), dtype=np.float32)
    
    xm, ym = x-mx, y-my
    
    r_num = K.cast(K.sum(tf.multiply(xm,ym)), dtype=np.float32)
    r_den = K.cast(K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym)))), dtype=np.float32)
    
    score = r_num / r_den
    return score
'''
def pearson(y_true, y_pred):
    return (tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true))
'''

def spearman(y_true, y_pred):
    from scipy.stats import spearmanr
    
    return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout = tf.float32) )

def dice_coef(
        y_true,
        y_pred,
        y_true_min=-0.5,
        unknown_coef=10
):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    mask = K.cast(
        K.greater_equal(y_true, y_true_min),
        dtype="float32"
    )
    intersection = K.sum(y_true * y_pred * mask)
    numerator = 2.0 * intersection + unknown_coef
    denominator = K.sum(y_true * mask) + K.sum(y_pred * mask) + unknown_coef
    return numerator / denominator


def tp(y_true, y_pred, pred_thresh=0.5):
    y_true = K.cast(K.flatten(y_true), dtype='float32')
    y_pred = K.cast(K.flatten(y_pred), dtype='float32')
    binary_preds = K.cast(K.greater_equal(y_pred, pred_thresh), dtype="float32")
    true_positives = K.cast(K.sum((K.clip(y_true * binary_preds, 0, 1))), dtype='float32')
    return true_positives


def tn(y_true, y_pred, pred_thresh=0.5):
    y_true = K.cast(K.flatten(y_true), dtype='float32')
    y_pred = K.cast(K.flatten(y_pred), dtype='float32')
    binary_preds = K.cast(K.greater_equal(y_pred, pred_thresh), dtype="float32")
    y_inv_true = K.cast(1.0 - y_true, dtype='float32')
    binary_inv_preds = K.cast(1.0 - binary_preds, dtype='float32')
    true_negatives = K.cast(K.sum((K.clip(y_inv_true * binary_inv_preds, 0, 1))), dtype="float32")
    return true_negatives


def fp(y_true, y_pred, pred_thresh=0.5):
    y_true = K.cast(K.flatten(y_true), dtype='float32')
    y_pred = K.cast(K.flatten(y_pred), dtype='float32')
    binary_preds = K.cast(K.greater_equal(y_pred, pred_thresh), dtype="float32")
    y_inv_true = K.cast(1.0 - y_true, dtype='float32')
    binary_inv_preds = K.cast(1.0 - binary_preds, dtype='float32')
    false_positives = K.cast(K.sum((K.clip(y_inv_true * binary_preds, 0, 1))), dtype="float32")
    return false_positives


def fn(y_true, y_pred, pred_thresh=0.5):
    y_true = K.cast(K.flatten(y_true), dtype='float32')
    y_pred = K.cast(K.flatten(y_pred), dtype='float32')
    binary_preds = K.cast(K.greater_equal(y_pred, pred_thresh), dtype="float32")
    y_inv_true = K.cast(1.0 - y_true, dtype='float32')
    binary_inv_preds = K.cast(1.0 - binary_preds, dtype='float32')
    false_negatives = K.cast(K.sum((K.clip(y_true * binary_inv_preds, 0, 1))), dtype="float32")
    return false_negatives


def acc(y_true, y_pred, pred_thresh=0.5):
    y_true = K.cast(K.flatten(y_true), dtype='float32')
    y_pred = K.cast(K.flatten(y_pred), dtype='float32')

    binary_preds = K.cast(K.greater_equal(y_pred, pred_thresh), dtype="float32")
    y_inv_true = K.cast(1.0 - y_true, dtype='float32')
    binary_inv_preds = K.cast(1.0 - binary_preds, dtype='float32')
    true_positives = K.cast(K.sum((K.clip(y_true * binary_preds, 0, 1))), dtype="float32")
    true_negatives = K.cast(K.sum((K.clip(y_inv_true * binary_inv_preds, 0, 1))), dtype="float32")
    false_positives = K.cast(K.sum((K.clip(y_inv_true * binary_preds, 0, 1))), dtype="float32")
    false_negatives = K.cast(K.sum((K.clip(y_true * binary_inv_preds, 0, 1))), dtype="float32")
    total = K.cast(true_positives + true_negatives + false_positives + false_negatives, dtype="float32")
    accuracy = K.cast((true_positives + true_negatives) / total, dtype="float32")
    # val = np.array([true_positives, true_negatives, false_positives, false_negatives, accuracy], dtype="float32")
    # conf_vector = K.constant(value= val, dtype='float32', name='conf_values')
    return (accuracy)


def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def get_layer(
        inbound_layer,
        filters,
        kernel_size,
        activation,
        padding,
        dilation_rate=1,
        skip_batch_norm=False,
        kernel_initializer='glorot_uniform',
        concat_layer=None,
        transpose_kernel_size=None,
        transpose_strides=None,
        n=2
):
    """
    Returns new layer without max pooling. If concat_layer,
    transpose_kernel_size and transpose_strides are provided
    run Conv1DTranspose and Concatenation. Optionally, you
    can skip batch normalization
    """
    for i in range(n):
        inbound_layer = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer
        )(inbound_layer)
        if not skip_batch_norm:
            inbound_layer = BatchNormalization()(inbound_layer)
    return inbound_layer


def get_dilated_cnn(
        output_activation,
        adam_learning_rate=DEFAULT_ADAM_LEARNING_RATE,
        adam_decay=DEFAULT_ADAM_DECAY,
        input_length=INPUT_LENGTH,
        input_channels=INPUT_CHANNELS,
        input_filters=INPUT_FILTERS,
        input_kernel_size=INPUT_KERNEL_SIZE,
        input_activation=INPUT_ACTIVATION,
        output_filters=OUTPUT_FILTERS,
        output_kernel_size=OUTPUT_KERNEL_SIZE,
        filters_scaling_factor=FILTERS_SCALING_FACTOR,
        dilation_rate=DILATION_RATE,
        output_length=OUTPUT_LENGTH,
        conv_blocks=CONV_BLOCKS,
        padding=PADDING,
        pool_size=POOL_SIZE,
        adam_beta_1=ADAM_BETA_1,
        adam_beta_2=ADAM_BETA_2,
        target_scale_factor=1,
        dense_b=False,
        weights=None
):
    """
    If weights are provided they will be loaded into created model
    """
    logging.debug("Building Dilated CNN model")

    # Inputs
    input_layer = Input(shape=(input_length, input_channels))

    # Temporary variables
    layer = input_layer  # redefined in encoder/decoder loops
    filters = input_filters  # redefined in encoder/decoder loops

    #logging.debug("Added inputs layer: " + "\n - " + str(layer))

    # Encoder
    all_layers = []
    for i in range(conv_blocks - 1):  # [0, 1, 2, 3, 4, 5]
        layer_dilation_rate = dilation_rate[i]
        layer = get_layer(
            inbound_layer=layer,  # input_layer is used wo MaxPooling1D
            filters=filters,
            kernel_size=input_kernel_size,
            activation=input_activation,
            padding=padding,
            dilation_rate=layer_dilation_rate,
            kernel_initializer=KERNEL_INITIALIZER
        )
        #logging.debug("Added convolution layer: " + str(i) + "\n - " + str(layer))
        # encoder_layers.append(layer)  # save all layers wo MaxPooling1D
        if i < conv_blocks - 1:  # need to update all except the last layers
            filters = round(filters * filters_scaling_factor)
            layer = MaxPooling1D(pool_size=pool_size, strides=pool_size)(layer)
        all_layers.append(layer)

    # Outputs
    layer_dilation_rate = dilation_rate[-1]
    if dense_b:
        output_layer = get_layer(
            inbound_layer=layer,
            filters=output_filters,
            kernel_size=output_kernel_size,
            activation=input_activation,
            padding=padding,
            dilation_rate=layer_dilation_rate,
            kernel_initializer=KERNEL_INITIALIZER,
            skip_batch_norm=True,
            n=1
        )
    else:
        output_layer = get_layer(
            inbound_layer=layer,
            filters=output_filters,
            kernel_size=output_kernel_size,
            activation=output_activation,
            padding=padding,
            dilation_rate=layer_dilation_rate,
            kernel_initializer=KERNEL_INITIALIZER,
            skip_batch_norm=True,
            n=1
        )

    # Depending on the output activation functions, model outputs need to be scaled appropriately
    output_layer = Flatten()(output_layer)
    if dense_b:
        output_layer = Dense(output_length, activation=output_activation, kernel_initializer='glorot_uniform')(
            output_layer)


    logging.debug("Added outputs layer: " + "\n - " + str(output_layer))

    # Model
    model = Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(
        optimizer=Adam(
            lr=adam_learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            decay=adam_decay
        ),
        loss=loss_function,
        metrics=[dice_coef]
    )

    logging.debug("Model compiled")

    if weights is not None:
        model.load_weights(weights)
        logging.debug("Weights loaded")

    return model

