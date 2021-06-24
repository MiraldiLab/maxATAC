from maxatac.utilities.system_tools import Mute

with Mute():  # hide stdout from loading the modules
    from keras import backend as K
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf


def configure_session(threads, number_GPU=0, reserved=0.05):
    # config = tf.ConfigProto(device_count={'GPU': number_GPU, 'CPU': threads})
    config = tf.compat.v1.ConfigProto()

    memory_fraction = 1 / float(threads) - reserved
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    set_session(tf.compat.v1.Session(config=config))
    K.set_image_data_format("channels_last")
