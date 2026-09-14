from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from .batch_logger import BatchLossLogger  # import your custom logger

def get_callbacks(model_location,
                  log_location,
                  tensor_board_log_dir,
                  monitor,
                  save_weights_only=False,
                  save_best_only=False,
                  append_log=False,
                  tensor_board_write_images=False,
                  tensor_board_write_graph=True,
                  batch_log_location=False,  # new argument
                  ):
    callbacks = [
        ModelCheckpoint(filepath=model_location,
                        save_weights_only=save_weights_only,
                        save_best_only=save_best_only,
                        monitor=monitor),
        CSVLogger(log_location,
                  separator=",",
                  append=append_log),
        TensorBoard(tensor_board_log_dir,
                    write_images=tensor_board_write_images,
                    write_graph=tensor_board_write_graph,
                    update_freq="batch")
    ]

    if batch_log_location:
        list = log_location.split("/")[:-1]
        list.append('ELK1_quant_batch_log.csv')
        batch_log_location = "/".join(list)
        callbacks.append(BatchLossLogger(batch_log_location))

    return callbacks
