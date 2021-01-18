import matplotlib.pyplot as plt
from keras.utils import plot_model
import sys
import random
from os import path
import pandas as pd
from maxatac.utilities.system_tools import get_dir, replace_extension, remove_tags, Mute

with Mute():  # hide stdout from loading the modules
    from maxatac.architectures.dcnn import (get_dilated_cnn, get_callbacks)


class MaxATACModel(object):
    """
    This class will initialize and fit a maxATAC model.
    """

    def __init__(self,
                 arch,
                 seed,
                 output_directory,
                 prefix,
                 number_of_filters,
                 kernel_size,
                 filter_scaling_factor,
                 threads,
                 training_monitor,
                 meta_path
                 ):
        """
        Initialize a maxATAC model
        
        :param arch: Neural network architecture to use
        :param seed: Random seed to use
        :param output_directory: Path to output directory
        :param prefix: Prefix to use for filename
        :param number_of_filters: The number of filters to use for input layer
        :param kernel_size: Kernel width in base pairs
        :param filter_scaling_factor: Scale the filter number by this factor each layer
        :param threads: Number of threads to use
        :param training_monitor: Metric to use to monitor model training
        :param meta_path: Path to the meta file associated with the run
        """
        self.arch = arch
        self.seed = seed
        self.output_directory = get_dir(output_directory)
        self.model_filename = prefix + "_{epoch}" + ".h5"
        self.results_location = path.join(self.output_directory, self.model_filename)
        self.log_location = replace_extension(remove_tags(self.results_location, "_{epoch}"), ".csv")
        self.tensor_board_log_dir = get_dir(path.join(self.output_directory, "tensorboard"))
        self.number_of_filters = number_of_filters
        self.kernel_size = kernel_size
        self.filter_scaling_factor = filter_scaling_factor
        self.training_monitor = training_monitor
        self.threads = threads
        self.training_history = ""

        # Set the random seed for the model
        random.seed(seed)

        self.meta_path = meta_path

        self.meta_dataframe = self.__import_meta()
        self.peak_paths = self.__get_peak_paths()
        self.cell_types = self.__get_cell_types()

        # Get the neural network model based on the specified model architecture
        if arch == "DCNN_V2":
            self.nn_model = get_dilated_cnn(
                input_filters=self.number_of_filters,
                input_kernel_size=self.kernel_size,
                filters_scaling_factor=self.filter_scaling_factor
            )

        else:
            sys.exit("Model Architecture not specified correctly. Please check")

    def __import_meta(self):
        """
        Import the meta file into a dataframe

        :return: Meta data in a pandas dataframe
        """
        return pd.read_csv(self.meta_path, sep='\t', header=0, index_col=None)

    def __get_cell_types(self):
        """
        Get the unique cell lines in the meta dataframe

        :return: A list of unique cell types
        """
        return self.meta_dataframe["Cell_Line"].unique().tolist()

    def __get_peak_paths(self):
        """
        Get the ATAC-seq and ChIP-seq peak paths from the meta data file and concatenate them

        :return: A list of unique paths to all ATAC-seq and ChIP-seq data
        """
        return self.meta_dataframe["ATAC_Peaks"].unique().tolist() + self.meta_dataframe["CHIP_Peaks"].unique().tolist()

    def fit_model(self, train_gen, val_gen, epochs, training_steps_per_epoch, validation_steps_per_epoch):
        """
        Fit the maxATAC model using the provided training and validation generators.

        :param validation_steps_per_epoch:
        :param training_steps_per_epoch:
        :param train_gen: The training data generator
        :param val_gen: The validation data generator
        :param epochs: Number of epochs to run the model for

        :return: Model training history
        """
        self.training_history = self.nn_model.fit_generator(generator=train_gen,
                                                            validation_data=val_gen,

                                                            epochs=epochs,
                                                            callbacks=get_callbacks(
                                                                model_location=self.results_location,
                                                                log_location=self.log_location,
                                                                tensor_board_log_dir=self.tensor_board_log_dir,
                                                                monitor=self.training_monitor
                                                            ),
                                                            steps_per_epoch=training_steps_per_epoch,
                                                            validation_steps=validation_steps_per_epoch,
                                                            use_multiprocessing=self.threads > 1,
                                                            workers=self.threads,
                                                            verbose=1
                                                            )

    def export_model_structure(self):
        """
        Export the model structure as a PDF.

        :return: Saves a PDF version of the model structure from Keras.
        """
        plot_model(model=self.nn_model,
                   show_shapes=True,
                   show_layer_names=True,
                   to_file=replace_extension(remove_tags(self.results_location,
                                                         "_{epoch}"
                                                         ),
                                             "_model_structure" + ".png"
                                             )
                   )

    def plot_metrics(self,
                     metric,
                     style="ggplot",
                     val_metric="",
                     title="",
                     y_label="",
                     suffix=""
                     ):
        """
        Plot the loss, accuracy, or dice coefficient training history.

        :param suffix:
        :param y_label:
        :param title:
        :param val_metric:
        :param metric: Metric to plot ("loss", "acc", "dice_coef")
        :param style: Style of plot to use. Default: "ggplot"

        :return: Saves a PDF image of the training history curves.
        """
        # Set the plot style to use
        plt.style.use(style)

        # Build the labels and names based on the metric of choice
        if metric == 'dice_coef':
            val_metric = "val_dice_coef"
            title = "Model Dice Coefficient"
            y_label = "Dice Coefficient"
            suffix = "_model_dice"

        elif metric == 'acc':
            val_metric = "val_acc"
            title = "Model Accuracy"
            y_label = "Accuracy"
            suffix = "_model_accuracy"

        elif metric == "loss":
            val_metric = "val_loss"
            title = "Model Loss"
            y_label = "Loss"
            suffix = "_model_loss"

        else:
            pass

        # Get the training X and Y values
        t_y = self.training_history.history[metric]
        t_x = [int(i) for i in range(1, len(t_y) + 1)]

        # Get the validation X and Y values
        v_y = self.training_history.history[val_metric]
        v_x = [int(i) for i in range(1, len(v_y) + 1)]

        # Plot the training and validation data
        plt.plot(t_x, t_y, marker='o')
        plt.plot(v_x, v_y, marker='o')

        # Set the x ticks based on the training x epochs.
        plt.xticks(t_x)

        # If the metric is accuracy or dice_coef set the ylim to (0,1)
        if metric == "acc" or metric == "dice_coef":
            plt.ylim(0, 1)

        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel("Epoch")

        plt.legend(["Training", "Validation"], loc="upper left")

        # Save figure based on the filename
        plt.savefig(
            replace_extension(
                remove_tags(self.results_location, "_{epoch}"),
                suffix + ".pdf"
            ),
            bbox_inches="tight"
        )

        # Close figure
        plt.close("all")
