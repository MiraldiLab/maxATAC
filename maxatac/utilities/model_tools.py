import matplotlib.pyplot as plt
from keras.utils import plot_model
import sys
import random
from os import path
import logging

from maxatac.utilities.system_tools import (
    get_dir,
    replace_extension,
    remove_tags,
    Mute
)

with Mute():  # hide stdout from loading the modules
    from maxatac.architectures.dcnn import (get_dilated_cnn, get_callbacks)

from maxatac.utilities.constants import TRAIN_MONITOR


class GetModel(object):
    """
    This is a class for training a maxATAC model

    Args
    ----
        seed (int, optional):
            Random seed to use.
        OutDir (str):
            Path to directory for storing results.
        prefix (str):
            Prefix string for building model name.
        arch (str):
            Architecture to use.
        FilterNumber (int):
            Number of filters to use in the input layer.
        KernelSize (int):
            Size of the kernel in base pairs of the input layer.
        FilterScalingFactor (float):
            Multiply the number of filters each layer by this.
        TrainMonitor (str):
            The statistic to use to monitor training.
        threads (int):
            Number of threads to use for training.

    Attributes
    ----------
        arch (str):
            Architecture to use.
        seed (int):
            Random state seed.
        OutDir (str):
            Output directory for storing results.
        model_filename (str):
            The model filename.
        results_location (str):
            Output directory and model filename.
        log_location (str):
            Path to save logs.
        tensor_board_log_dir (str):
            Path to tensor board log.
        FilterNumber (int):
            Number of filters to use in the input layer.
        KernelSize (int):
            Size of the kernel in base pairs of the input layer.
        LRate (float):
            Adam learning rate.
        decay (float):
            Adam decay rate.
        FilterScalingFactor (float):
            Multiply the number of filters each layer by this.
        batches (int):
            The number of batches to use for training model.
        epochs (int):
            The number of epochs to train model for.
        nn_model (obj):
            The neural network model to be used for training.
        TrainMonitor (str):
            The statistic to use to monitor training.
        threads (int):
            Number of threads to use for training.

    Methods
    -------
        _get_DCNN
            Get a DCNN model
        FitModel
            Fit the model using the training data
    """

    def __init__(self,
                 arch,
                 seed,
                 OutDir,
                 prefix,
                 FilterNumber,
                 KernelSize,
                 FilterScalingFactor,
                 threads,
                 TrainMonitor=TRAIN_MONITOR
                 ):
        self.arch = arch
        self.seed = seed
        self.OutDir = get_dir(OutDir)
        self.model_filename = prefix + "_{epoch}" + ".h5"
        self.results_location = path.join(self.OutDir, self.model_filename)
        self.log_location = replace_extension(remove_tags(self.results_location, "_{epoch}"), ".csv")
        self.tensor_board_log_dir = get_dir(path.join(self.OutDir, "tensorboard"))
        self.FilterNumber = FilterNumber
        self.KernelSize = KernelSize
        self.FilterScalingFactor = FilterScalingFactor
        self.TrainMonitor = TrainMonitor
        self.threads = threads
        self.training_history = ""

        random.seed(seed)

        if arch == "DCNN_V2":
            self.nn_model = get_dilated_cnn(
                input_filters=self.FilterNumber,
                input_kernel_size=self.KernelSize,
                filters_scaling_factor=self.FilterScalingFactor
            )

        else:
            sys.exit("Model Architecture not specified correctly. Please check")

    def FitModel(self, train_gen, val_gen, train_batches, epochs):
        """
        Fit the model with the specific number of batches and epochs

        Args
        ----
            batches (int):
                The number of batches to use for training model.
            epochs (int):
                The number of epochs to train model for.

        Attributes
        ----------
            training_history (obj):
                A history of model training and the parameters
        """
        self.training_history = self.nn_model.fit_generator(
            generator=train_gen,
            validation_data=val_gen,
            steps_per_epoch=train_batches,
            epochs=epochs,
            callbacks=get_callbacks(
                model_location=self.results_location,
                log_location=self.log_location,
                tensor_board_log_dir=self.tensor_board_log_dir,
                monitor=self.TrainMonitor
            ),
            use_multiprocessing=self.threads > 1,
            workers=self.threads,
            verbose=1
        )

    def PlotResults(self):
        export_model_structure(self.nn_model, self.results_location)
        export_model_loss(self.training_history, self.results_location)
        export_model_dice(self.training_history, self.results_location)
        export_model_accuracy(self.training_history, self.results_location)

        logging.error("Results are saved to: " + self.results_location)


def export_model_structure(model, file_location, suffix="_model_structure", ext=".pdf", skip_tags="_{epoch}"):
    plot_model(
        model=model,
        show_shapes=True,
        show_layer_names=True,
        to_file=replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        )
    )


def export_model_loss(history, file_location, suffix="_model_loss", ext=".pdf", style="ggplot", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)

    t_y = history.history['loss']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_loss"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    plt.plot(t_x, t_y, marker='o')
    plt.plot(v_x, v_y, marker='o')

    plt.xticks(t_x)

    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper right")

    plt.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )

    plt.close("all")


def export_model_dice(history, file_location, suffix="_model_dice", ext=".pdf", style="ggplot", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)

    t_y = history.history['dice_coef']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_dice_coef"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    plt.plot(t_x, t_y, marker='o')
    plt.plot(v_x, v_y, marker='o')

    plt.xticks(t_x)
    plt.ylim(0, 1)

    plt.title("Model Dice Coefficient")
    plt.ylabel("Dice Coefficient")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper left")

    plt.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )

    plt.close("all")


def export_model_accuracy(history, file_location, suffix="_model_accuracy", ext=".pdf", style="ggplot", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)

    t_y = history.history['acc']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_acc"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    plt.plot(t_x, t_y, marker='o')
    plt.plot(v_x, v_y, marker='o')

    plt.xticks(t_x)
    plt.ylim(0, 1)

    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper left")

    plt.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )

    plt.close("all")
