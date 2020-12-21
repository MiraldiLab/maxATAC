import tensorflow as tf

from maxatac.utilities.helpers import Mute
with Mute():  # hide stdout from loading the modules
    from keras import backend as K
    from keras.backend.tensorflow_backend import set_session


def configure_session(threads, reserved=0.05):
    config = tf.ConfigProto()
    memory_fraction = 1/float(threads) - reserved
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    set_session(tf.Session(config=config))
    K.set_image_data_format("channels_last")
