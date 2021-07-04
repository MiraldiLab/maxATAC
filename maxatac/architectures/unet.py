import logging
from maxatac.utilities.system_tools import Mute

with Mute():
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        concatenate,
        Conv1D,
        MaxPooling1D,
        Conv2DTranspose,
        Lambda,
        BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K


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


# TODO should be refactored into something easy to understand without lambda
def Conv1DTranspose(inbound_layer, filters, kernel_size, strides, padding):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(inbound_layer)
    x = Conv2DTranspose(
        filters=filters,
        kernel_size=(kernel_size, 1),
        strides=(strides, 1),
        padding=padding
    )(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x


def get_layer(
        inbound_layer,
        filters,
        kernel_size,
        activation,
        padding,
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
    if concat_layer is not None \
            and transpose_kernel_size is not None \
            and transpose_strides is not None:
        inbound_layer = concatenate(
            [
                Conv1DTranspose(
                    inbound_layer=inbound_layer,
                    filters=filters,
                    kernel_size=transpose_kernel_size,
                    strides=transpose_strides,
                    padding=padding
                ),
                concat_layer
            ],
            axis=2  # using "filter" axis - (batch, new_steps, filters)
        )
    for i in range(n):
        inbound_layer = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding
        )(inbound_layer)
        if not skip_batch_norm:
            inbound_layer = BatchNormalization()(inbound_layer)
    return inbound_layer


def get_unet(
        input_length,
        input_channels,
        input_filters,
        input_kernel_size,
        input_activation,
        output_filters,
        output_kernel_size,
        output_activation,
        filters_scaling_factor,
        conv_blocks,
        padding,
        pool_size,
        adam_learning_rate,
        adam_beta_1,
        adam_beta_2,
        adam_decay,
        weights=None
):
    """
    If weights are provided they will be loaded into created model
    """
    logging.debug("Building Unet model")

    # Inputs
    input_layer = Input(shape=(input_length, input_channels))

    # Temporary variables
    layer = input_layer  # redefined in encoder/decoder loops
    filters = input_filters  # redefined in encoder/decoder loops

    logging.debug("Added inputs layer: " + "\n - " + str(layer))

    # Encoder
    encoder_layers = []
    for i in range(conv_blocks):  # [0, 1, 2, 3, 4, 5]
        layer = get_layer(
            inbound_layer=layer,  # input_layer is used wo MaxPooling1D
            filters=filters,
            kernel_size=input_kernel_size,
            activation=input_activation,
            padding=padding
        )
        logging.debug("Added encoder layer: " + str(i) + "\n - " + str(layer))
        encoder_layers.append(layer)  # save all layers wo MaxPooling1D
        if i < conv_blocks - 1:  # need to update all except the last layers
            filters = round(filters * filters_scaling_factor)
            layer = MaxPooling1D(pool_size=pool_size, strides=pool_size)(layer)

    # Decoder
    for i in range(conv_blocks - 2, -1, -1):  # [4, 3, 2, 1, 0]
        filters = round(filters / filters_scaling_factor)
        layer = get_layer(
            inbound_layer=layer,
            concat_layer=encoder_layers[i],
            transpose_kernel_size=pool_size,
            transpose_strides=pool_size,
            filters=filters,
            kernel_size=input_kernel_size,
            activation=input_activation,
            padding=padding
        )
        logging.debug("Added decoder layer: " + str(i) + "\n - " + str(layer))

    # Outputs
    output_layer = get_layer(
        inbound_layer=layer,
        filters=output_filters,
        kernel_size=output_kernel_size,
        activation=output_activation,
        padding=padding,
        skip_batch_norm=True,
        n=1
    )

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


