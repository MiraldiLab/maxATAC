from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
import os

from .batch_logger import BatchLossLogger

def get_callbacks(model_location,
                  log_location,
                  tensor_board_log_dir,
                  monitor,
                  save_weights_only=False,
                  save_best_only=False,
                  append_log=False,
                  tensor_board_write_images=False,
                  tensor_board_write_graph=True,
                  batch_log_location=False,  # True: also write a per-batch loss log next to log_location
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
        # Name the batch log after the run's epoch log, e.g. <prefix>.csv -> <prefix>_batch.csv
        stem, ext = os.path.splitext(log_location)
        batch_log_location = stem + "_batch" + (ext or ".csv")
        callbacks.append(BatchLossLogger(batch_log_location))

    return callbacks
