from maxatac.utilities.system_tools import Mute

from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

with Mute():  # hide stdout from loading the modules
    from tensorflow.keras import backend as K
    import tensorflow as tf


def configure_session(threads, number_GPU=0, reserved=0.05):
    # config = tf.ConfigProto(device_count={'GPU': number_GPU, 'CPU': threads})
    config = ConfigProto()
    memory_fraction = 0.95
    #memory_fraction = 1 / float(threads) - reserved
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction

    session = tf.compat.v1.Session(config=config)

    K.set_image_data_format("channels_last")
