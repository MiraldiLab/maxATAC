import numpy as np
import os
from tensorflow.keras.callbacks import Callback

class SavePredictionsCallback(Callback):
    def __init__(self, save_dir, prefix="train", save_format="npz"):
        super().__init__()
        self.save_dir = save_dir
        self.prefix = prefix
        self.save_format = save_format
        os.makedirs(self.save_dir, exist_ok=True)
        self.reset_storage()

    def reset_storage(self):
        self.all_y_true = []
        self.all_y_pred = []

    def on_epoch_begin(self, epoch, logs=None):
        self.reset_storage()
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        # Ensure model has access to y_true/y_pred
        x, y_true = self.model._last_train_batch
        y_pred = self.model(x, training=False)
        self.all_y_true.append(y_true.numpy())
        self.all_y_pred.append(y_pred.numpy())

    def on_epoch_end(self, epoch, logs=None):
        y_true_all = np.concatenate(self.all_y_true, axis=0)
        y_pred_all = np.concatenate(self.all_y_pred, axis=0)
        
        filename = f"{self.prefix}_epoch{self.epoch:02d}.{self.save_format}"
        save_path = os.path.join(self.save_dir, filename)

        if self.save_format == "npz":
            np.savez(save_path, y_true=y_true_all, y_pred=y_pred_all)
        elif self.save_format == "csv":
            np.savetxt(save_path.replace(".csv", "_true.csv"), y_true_all, delimiter=",")
            np.savetxt(save_path.replace(".csv", "_pred.csv"), y_pred_all, delimiter=",")
        else:
            raise ValueError("Unsupported save_format: use 'npz' or 'csv'")

