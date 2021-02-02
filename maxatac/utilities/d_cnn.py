import logging
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import (
    Input,
    concatenate,
    Conv1D,
    MaxPooling1D,
    Conv2DTranspose,
    Lambda,
    BatchNormalization
)
from keras.layers.core import Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras



from maxatac.utilities.constants import (
    INPUT_CHANNELS,
    MIN_PREDICTION, # min prediction value to report in output
    BATCH_SIZE,
    CHR_POOL_SIZE,
    FLANK_LENGTH,
    BP_ORDER,
    PHASES,
    INPUT_FILTERS,
    INPUT_KERNEL_SIZE,
    INPUT_LENGTH,
    INPUT_ACTIVATION,
    PADDING,
    FILTERS_SCALING_FACTOR,
    CONV_BLOCKS,
    DILATION_RATE,
    OUTPUT_FILTERS,
    OUTPUT_KERNEL_SIZE,
    BINARY_OUTPUT_ACTIVATION,
    QUANT_OUTPUT_ACTIVATION,
    POOL_SIZE,
    ADAM_BETA_1,
    ADAM_BETA_2,
    TRAIN_SCALE_SIGNAL,
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
    return numerator/denominator

def tp(y_true, y_pred, pred_thresh=0.5):
    y_true = K.cast(K.flatten(y_true), dtype='float32')
    y_pred = K.cast(K.flatten(y_pred), dtype='float32')
    binary_preds = K.cast(K.greater_equal(y_pred, pred_thresh),dtype="float32")
    true_positives = K.cast(K.sum((K.clip (y_true * binary_preds, 0, 1))), dtype='float32')
    return true_positives

def tn(y_true, y_pred, pred_thresh=0.5):
    y_true = K.cast(K.flatten(y_true), dtype='float32')
    y_pred = K.cast(K.flatten(y_pred), dtype='float32')
    binary_preds = K.cast(K.greater_equal(y_pred, pred_thresh),dtype="float32")
    y_inv_true = K.cast(1.0 - y_true, dtype='float32')
    binary_inv_preds = K.cast(1.0 - binary_preds, dtype='float32')
    true_negatives = K.cast(K.sum((K.clip(y_inv_true * binary_inv_preds, 0, 1))), dtype="float32")
    return true_negatives
    
def fp(y_true, y_pred, pred_thresh=0.5):
    y_true = K.cast(K.flatten(y_true), dtype='float32')
    y_pred = K.cast(K.flatten(y_pred), dtype='float32')
    binary_preds = K.cast(K.greater_equal(y_pred, pred_thresh),dtype="float32")
    y_inv_true = K.cast(1.0 - y_true, dtype='float32')
    binary_inv_preds = K.cast(1.0 - binary_preds, dtype='float32')
    false_positives = K.cast(K.sum((K.clip(y_inv_true * binary_preds, 0, 1))), dtype="float32")
    return false_positives
 
def fn(y_true, y_pred, pred_thresh=0.5):
    y_true = K.cast(K.flatten(y_true), dtype='float32')
    y_pred = K.cast(K.flatten(y_pred), dtype='float32')
    binary_preds = K.cast(K.greater_equal(y_pred, pred_thresh),dtype="float32")
    y_inv_true = K.cast(1.0 - y_true, dtype='float32')
    binary_inv_preds = K.cast(1.0 - binary_preds, dtype='float32')
    false_negatives = K.cast(K.sum((K.clip(y_true * binary_inv_preds, 0, 1))), dtype="float32")
    return false_negatives

def acc(y_true, y_pred, pred_thresh=0.5):
    y_true = K.cast(K.flatten(y_true), dtype='float32')
    y_pred = K.cast(K.flatten(y_pred), dtype='float32')

    binary_preds = K.cast(K.greater_equal(y_pred, pred_thresh),dtype="float32")
    y_inv_true = K.cast(1.0 - y_true, dtype='float32')
    binary_inv_preds = K.cast(1.0 - binary_preds, dtype='float32')
    true_positives = K.cast(K.sum((K.clip(y_true * binary_preds, 0, 1))), dtype="float32")
    true_negatives = K.cast(K.sum((K.clip(y_inv_true * binary_inv_preds, 0, 1))), dtype="float32")
    false_positives = K.cast(K.sum((K.clip(y_inv_true * binary_preds, 0, 1))), dtype="float32")
    false_negatives = K.cast(K.sum((K.clip(y_true * binary_inv_preds, 0, 1))), dtype="float32")
    total = K.cast(true_positives + true_negatives + false_positives + false_negatives, dtype="float32")
    accuracy = K.cast((true_positives + true_negatives)/total, dtype="float32")
    #val = np.array([true_positives, true_negatives, false_positives, false_negatives, accuracy], dtype="float32")
    #conf_vector = K.constant(value= val, dtype='float32', name='conf_values')
    return (accuracy)

def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def get_layer(
    inbound_layer,
    filters,
    kernel_size,
    activation,
    padding,
    dilation_rate=1,
    skip_batch_norm=False,
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
            dilation_rate=dilation_rate
        )(inbound_layer)
        if not skip_batch_norm:
            inbound_layer = BatchNormalization()(inbound_layer)
    return inbound_layer


def get_dilated_cnn(
    input_length,
    input_channels,
    input_filters,
    input_kernel_size,
    input_activation,
    output_filters,
    output_kernel_size,
    output_activation,
    filters_scaling_factor,
    dilation_rate,
    conv_blocks,
    padding,
    pool_size,
    adam_learning_rate,
    adam_beta_1,
    adam_beta_2,
    adam_decay,
    quant=False,
    weights=None
):
    """
    If weights are provided they will be loaded into created model
    """
    logging.debug("Building Dilated CNN model")

    # Inputs
    input_layer = Input(shape=(input_length, input_channels))

    # Temporary variables
    layer = input_layer                   # redefined in encoder/decoder loops
    filters = input_filters               # redefined in encoder/decoder loops

    logging.debug("Added inputs layer: " + "\n - " + str(layer))

    # Encoder
    all_layers = []
    for i in range(conv_blocks-1):          # [0, 1, 2, 3, 4, 5]
        layer_dilation_rate = dilation_rate[i]
        layer = get_layer(
                    inbound_layer=layer,  # input_layer is used wo MaxPooling1D
                    filters=filters,
                    kernel_size=input_kernel_size,
                    activation=input_activation,
                    padding=padding,
                    dilation_rate= layer_dilation_rate
        )
        logging.debug("Added convolution layer: " + str(i) + "\n - " + str(layer))
       #encoder_layers.append(layer)  # save all layers wo MaxPooling1D
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
    if not quant:
        model.compile(
            optimizer=Adam(
                lr=adam_learning_rate,
                beta_1=adam_beta_1,
                beta_2=adam_beta_2,
                decay=adam_decay
            ),
            loss=loss_function,
            metrics=[dice_coef, 'accuracy']
        )
    else:
        mse = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error") #May wnat to change Reduction methods possibly
        model.compile(
            optimizer=Adam(
                lr=adam_learning_rate,
                beta_1=adam_beta_1,
                beta_2=adam_beta_2,
                decay=adam_decay
            ),
            loss=mse,
            metrics=[mse, coeff_determination] #tf.keras.metrics.RootMeanSquaredError()
        )


    logging.debug("Model compiled")

    if weights is not None:
        model.load_weights(weights)
        logging.debug("Weights loaded")

    return model


def get_callbacks(
    location,
    monitor,
    save_weights_only=False,
    save_best_only=True
):
    callbacks = [
        ModelCheckpoint(
            filepath=location,
            save_weights_only=save_weights_only,
            save_best_only=save_best_only,
            monitor=monitor
        )
    ]
    return callbacks

'''
start =1
model = get_dilated_cnn(
    input_length=INPUT_LENGTH,
    input_channels=INPUT_CHANNELS,
    input_filters=INPUT_FILTERS,
    input_kernel_size=INPUT_KERNEL_SIZE,
    input_activation=INPUT_ACTIVATION,
    output_filters=OUTPUT_FILTERS,
    output_kernel_size=OUTPUT_KERNEL_SIZE,
    output_activation=OUTPUT_ACTIVATION,
    filters_scaling_factor=FILTERS_SCALING_FACTOR,
    dilation_rate=DILATION_RATE,
    conv_blocks=CONV_BLOCKS,
    padding=PADDING,
    pool_size=POOL_SIZE,
    adam_learning_rate=DEFAULT_ADAM_LEARNING_RATE,
    adam_beta_1=ADAM_BETA_1,
    adam_beta_2=ADAM_BETA_2,
    adam_decay=DEFAULT_ADAM_DECAY,
    weights=None)

debug = 1
'''