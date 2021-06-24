import logging

from maxatac.utilities.constants import INPUT_LENGTH, INPUT_CHANNELS, DNA_INPUT_CHANNELS, ATAC_INPUT_CHANNELS, \
    INPUT_FILTERS, INPUT_KERNEL_SIZE, INPUT_ACTIVATION, KERNEL_INITIALIZER, OUTPUT_FILTERS, OUTPUT_KERNEL_SIZE, \
    FILTERS_SCALING_FACTOR, DILATION_RATE, OUTPUT_LENGTH, PURE_CONV_LAYERS, CONV_BLOCKS, PADDING, POOL_SIZE, \
    DEFAULT_ADAM_LEARNING_RATE, ADAM_BETA_1, ADAM_BETA_2, DEFAULT_ADAM_DECAY
from maxatac.utilities.system_tools import Mute

with Mute():
    import tensorflow as tf
    from keras.models import Model
    from keras.layers import (
        Input,
        Concatenate,
        Conv1D,
        MaxPooling1D,
        Lambda,
        BatchNormalization,
        Dense,
        Add,
        Activation,
        Flatten
    )
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint
    from keras import backend as K


def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


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


# 1. Make a Model Class that implements DNA branch
# 2. Make a Model Class that implements ATAC branch
# 3. Finally make a Model Class that implements the concatenation and downstream side of model


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


def res_conv_block(
        inbound_layer,
        filters,
        kernel_size,
        activation,
        padding,
        kernel_initializer,
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
    identity = Conv1D(filters=filters,
                      kernel_size=1,
                      activation=activation,
                      padding=padding,
                      dilation_rate=1,
                      kernel_initializer=kernel_initializer
                      )(inbound_layer)
    # identity = BatchNormalization()(identity)

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

    add_layer = Add()([inbound_layer, identity])
    add_layer = BatchNormalization()(add_layer)
    res_output = Activation(activation)(add_layer)
    return res_output


def conv_block(
        inbound_layer,
        filters,
        kernel_size,
        activation,
        padding,
        kernel_initializer,
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
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer
        )(inbound_layer)
        if not skip_batch_norm:
            inbound_layer = BatchNormalization()(inbound_layer)
    return inbound_layer


def MM_DCNN_V2(output_activation,
               input_length=INPUT_LENGTH,
               input_channels=INPUT_CHANNELS,
               dna_input_channels=DNA_INPUT_CHANNELS,
               atac_input_channels=ATAC_INPUT_CHANNELS,
               input_filters=INPUT_FILTERS,
               input_kernel_size=INPUT_KERNEL_SIZE,
               input_activation=INPUT_ACTIVATION,
               kernel_initializer=KERNEL_INITIALIZER,
               output_filters=OUTPUT_FILTERS,
               output_kernel_size=OUTPUT_KERNEL_SIZE,
               filters_scaling_factor=FILTERS_SCALING_FACTOR,
               dilation_rate=DILATION_RATE,
               output_length=OUTPUT_LENGTH,
               pure_conv_layers=PURE_CONV_LAYERS,
               conv_blocks=CONV_BLOCKS,
               padding=PADDING,
               pool_size=POOL_SIZE,
               adam_learning_rate=DEFAULT_ADAM_LEARNING_RATE,
               adam_beta_1=ADAM_BETA_1,
               adam_beta_2=ADAM_BETA_2,
               adam_decay=DEFAULT_ADAM_DECAY,
               quant=False,
               target_scale_factor=1,
               dense_b=False,
               weights=None,
               res_conn=False,
               chanDim=-1
               ):
    if res_conn:
        block_fn = res_conv_block
    else:
        block_fn = conv_block

    input_layer = Input(shape=(input_length, input_channels))
    dna_input_layer = Lambda(lambda x: x[:, :, :dna_input_channels])(input_layer)
    atac_input_layer = Lambda(lambda x: x[:, :, dna_input_channels:])(input_layer)
    # dna_input_layer = Input(shape=(input_length, dna_input_channels))

    # atac_input_layer = Input(shape=(input_length, atac_input_channels))

    dna_layer = dna_input_layer
    atac_layer = atac_input_layer
    filters = input_filters

    for i in range(pure_conv_layers):
        dna_layer = block_fn(inbound_layer=dna_layer,
                             filters=filters,
                             kernel_size=input_kernel_size,
                             activation=input_activation,
                             padding=padding,
                             dilation_rate=1,
                             kernel_initializer=kernel_initializer
                             )
        atac_layer = block_fn(inbound_layer=atac_layer,
                              filters=filters,
                              kernel_size=input_kernel_size,
                              activation=input_activation,
                              padding=padding,
                              dilation_rate=1,
                              kernel_initializer=kernel_initializer
                              )
        filters = round(filters * filters_scaling_factor)

    for i in range(conv_blocks - 1):  # [0, 1, 2, 3, 4, 5]
        layer_dilation_rate = dilation_rate[i]
        dna_layer = block_fn(inbound_layer=dna_layer,
                             filters=filters,
                             kernel_size=input_kernel_size,
                             activation=input_activation,
                             padding=padding,
                             dilation_rate=layer_dilation_rate,
                             kernel_initializer=kernel_initializer
                             )
        atac_layer = block_fn(inbound_layer=atac_layer,
                              filters=filters,
                              kernel_size=input_kernel_size,
                              activation=input_activation,
                              padding=padding,
                              dilation_rate=layer_dilation_rate,
                              kernel_initializer=kernel_initializer
                              )
        if i < conv_blocks - 1:  # need to update all except the last layers
            filters = round(filters * filters_scaling_factor)
            dna_layer = MaxPooling1D(pool_size=pool_size, strides=pool_size)(dna_layer)
            atac_layer = MaxPooling1D(pool_size=pool_size, strides=pool_size)(atac_layer)

    layer_dilation_rate = dilation_rate[-1]
    penultimate_layer = Concatenate(axis=1)([dna_layer, atac_layer])

    penultimate_layer = block_fn(inbound_layer=penultimate_layer,
                                 filters=int(penultimate_layer.shape[-1]),
                                 kernel_size=input_kernel_size,
                                 activation=input_activation,
                                 padding=padding,
                                 dilation_rate=layer_dilation_rate,
                                 kernel_initializer=kernel_initializer
                                 )
    penultimate_layer = MaxPooling1D(pool_size=pool_size, strides=pool_size)(penultimate_layer)

    # concat_model = Model(inputs=[dna_input_layer, atac_input_layer], outputs=[penultimate_layer])

    layer_dilation_rate = dilation_rate[-1]
    output_layer = conv_block(
        inbound_layer=penultimate_layer,
        filters=output_filters,
        kernel_size=output_kernel_size,
        activation=input_activation,
        padding=padding,
        dilation_rate=layer_dilation_rate,
        kernel_initializer=kernel_initializer,
        skip_batch_norm=True,
        n=1
    )

    output_layer = Flatten()(output_layer)
    if dense_b:
        output_layer = Dense(output_length, activation=output_activation, kernel_initializer='glorot_uniform')(
            output_layer)

    if quant and output_activation in ["sigmoid"]:
        output_layer = Lambda(lambda x: x * target_scale_factor, name='Target_Scale_Layer')(output_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    # model = Model(inputs=[dna_input_layer, atac_input_layer], outputs=[output_layer])
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
        mse = tf.keras.losses.MeanSquaredError(reduction="auto",
                                               name="mean_squared_error")  # May wnat to change Reduction methods possibly
        model.compile(
            optimizer=Adam(
                lr=adam_learning_rate,
                beta_1=adam_beta_1,
                beta_2=adam_beta_2,
                decay=adam_decay
            ),
            loss=mse,
            metrics=[mse, coeff_determination]  # tf.keras.metrics.RootMeanSquaredError()
        )

    logging.debug("Model compiled")

    if weights is not None:
        model.load_weights(weights)
        logging.debug("Weights loaded")

    return model

# start =1
# model = MM_DCNN_V2(
#     input_length=INPUT_LENGTH,
#     input_channels=INPUT_CHANNELS,
#     dna_input_channels=DNA_INPUT_CHANNELS,
#     atac_input_channels=ATAC_INPUT_CHANNELS,
#     input_filters=INPUT_FILTERS,
#     input_kernel_size=INPUT_KERNEL_SIZE,
#     input_activation=INPUT_ACTIVATION,
#     kernel_initializer=KERNEL_INITIALIZER,
#     output_filters=OUTPUT_FILTERS,
#     output_kernel_size=OUTPUT_KERNEL_SIZE,
#     output_activation='sigmoid',
#     filters_scaling_factor=FILTERS_SCALING_FACTOR,
#     dilation_rate=DILATION_RATE,
#     output_length=OUTPUT_LENGTH,
#     pure_conv_layers=PURE_CONV_LAYERS,
#     conv_blocks=CONV_BLOCKS,
#     padding=PADDING,
#     pool_size=POOL_SIZE,
#     adam_learning_rate=DEFAULT_ADAM_LEARNING_RATE,
#     adam_beta_1=ADAM_BETA_1,
#     adam_beta_2=ADAM_BETA_2,
#     adam_decay=DEFAULT_ADAM_DECAY,
#     quant=True,
#     target_scale_factor=10,
#     dense_b=True,
#     weights=None)
# debug = 1
