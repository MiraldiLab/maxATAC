import csv
from tensorflow.keras.callbacks import Callback
'''
Per Batch Logger for values. Needs to be re-checked. 
'''
class BatchLossLogger(Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.epoch = 0
        self.fieldnames = None
        self.writer = None

    def on_train_begin(self, logs=None):
        self.file = open(self.log_file, 'w', newline='')
        self.writer = None  # delay until first batch

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs = logs.copy()
        logs['epoch'] = self.epoch
        logs['batch'] = batch

        if self.writer is None:
            self.fieldnames = ['epoch', 'batch'] + list(logs.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()

        self.writer.writerow(logs)

    def on_train_end(self, logs=None):
        self.file.close()

