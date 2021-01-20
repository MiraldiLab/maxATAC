import tensorflow as tf

import logging
from keras.models import Model
from keras.layers.core import Reshape
from keras.optimizers import Adam
from keras import backend as K

from keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization
)

from keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    TensorBoard
)

from maxatac.utilities.constants import (
    INPUT_CHANNELS,
    INPUT_LENGTH,
    INPUT_ACTIVATION,
    PADDING,
    CONV_BLOCKS,
    DILATION_RATE,
    OUTPUT_FILTERS,
    OUTPUT_KERNEL_SIZE,
    OUTPUT_ACTIVATION,
    POOL_SIZE,
    ADAM_BETA_1,
    ADAM_BETA_2,
    DEFAULT_ADAM_LEARNING_RATE,
    DEFAULT_ADAM_DECAY
)


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
        -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred),
        K.greater_equal(y_true, y_true_min)
    )

    return tf.reduce_mean(losses)


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

    return accuracy


def get_layer(
        inbound_layer,
        filters,
        kernel_size,
        activation,
        padding,
        dilation_rate=1,
        skip_batch_norm=False,
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
            dilation_rate=dilation_rate
        )(inbound_layer)

        if not skip_batch_norm:
            inbound_layer = BatchNormalization()(inbound_layer)

    return inbound_layer


def get_dilated_cnn(
        input_filters,
        input_kernel_size,
        filters_scaling_factor,
        adam_learning_rate=DEFAULT_ADAM_LEARNING_RATE,
        adam_decay=DEFAULT_ADAM_DECAY,
        input_length=INPUT_LENGTH,
        input_channels=INPUT_CHANNELS,
        input_activation=INPUT_ACTIVATION,
        output_filters=OUTPUT_FILTERS,
        output_kernel_size=OUTPUT_KERNEL_SIZE,
        output_activation=OUTPUT_ACTIVATION,
        conv_blocks=CONV_BLOCKS,
        padding=PADDING,
        pool_size=POOL_SIZE,
        adam_beta_1=ADAM_BETA_1,
        adam_beta_2=ADAM_BETA_2,
        dilation_rate=DILATION_RATE,
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

    logging.debug("Added inputs layer: " + "\n - " + str(layer))

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
            dilation_rate=layer_dilation_rate
        )

        logging.debug("Added convolution layer: " + str(i) + "\n - " + str(layer))

        if i < conv_blocks - 1:  # need to update all except the last layers
            filters = round(filters * filters_scaling_factor)
            layer = MaxPooling1D(pool_size=pool_size, strides=pool_size)(layer)
        all_layers.append(layer)

    # Outputs
    layer_dilation_rate = dilation_rate[-1]

    output_layer = get_layer(
        inbound_layer=layer,
        filters=output_filters,
        kernel_size=output_kernel_size,
        activation=output_activation,
        padding=padding,
        dilation_rate=layer_dilation_rate,
        skip_batch_norm=True,
        n=1
    )
    newdim = tuple([x for x in output_layer.shape.as_list() if x != 1 and x is not None])

    output_layer = Reshape(newdim)(output_layer)

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
        loss="binary_crossentropy",
        metrics=[dice_coef,
                 'binary_accuracy',
                 "accuracy"]
    )

    logging.debug("Model compiled")

    if weights is not None:
        model.load_weights(weights)
        logging.debug("Weights loaded")

    return model


def get_callbacks(
        model_location,
        log_location,
        tensor_board_log_dir,
        monitor,
        save_weights_only=False,
        save_best_only=False,
        append_log=False,
        tensor_board_write_images=False,
        tensor_board_write_graph=True,
):
    callbacks = [
        ModelCheckpoint(
            filepath=model_location,
            save_weights_only=save_weights_only,
            save_best_only=save_best_only,
            monitor=monitor
        ),
        CSVLogger(log_location, separator=",", append=append_log),
        TensorBoard(
            tensor_board_log_dir,
            write_images=tensor_board_write_images,
            write_graph=tensor_board_write_graph,
            update_freq="batch"
        )
    ]
    return callbacks
