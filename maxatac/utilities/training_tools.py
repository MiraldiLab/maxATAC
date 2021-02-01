import random

import keras
import pandas as pd
import numpy as np

from maxatac.utilities.genome_tools import (build_chrom_sizes_dict,
                                            get_input_matrix,
                                            load_bigwig,
                                            load_2bit,
                                            get_target_matrix)

import matplotlib.pyplot as plt
from keras.utils import plot_model
import sys
from os import path
from maxatac.utilities.system_tools import get_dir, replace_extension, remove_tags, Mute

from maxatac.utilities.roi_tools import RandomRegionsPool, import_meta

with Mute():  # hide stdout from loading the modules
    from maxatac.architectures.dcnn import (get_dilated_cnn, get_callbacks)


class TrainingDataGenerator(object):
    """
    Generate batches of examples for the chromosomes of interest

    This generator will create individual batches of examples from a pool of regions of interest and/or a pool of
    random regions from the genome. This class object helps keep track of all of the required inputs and how they are
    processed.

    This generator expects a meta_table with the header:

    TF | Cell Type | ATAC signal | ChIP signal | ATAC peaks | ChIP peaks
    """

    def __init__(self,
                 meta_dataframe,
                 chromosomes,
                 sequence,
                 batch_size,
                 region_length,
                 random_ratio,
                 chromosome_sizes,
                 bp_resolution,
                 input_channels,
                 scale_signal,
                 roi_dataframe,
                 preferences,
                 cell_types
                 ):
        self.meta_dataframe = meta_dataframe
        self.chromosomes = chromosomes
        self.region_length = region_length
        self.random_ratio = random_ratio
        self.input_channels = input_channels
        self.bp_resolution = bp_resolution
        self.batch_size = batch_size
        self.sequence = sequence
        self.region_length = region_length
        self.scale_signal = scale_signal
        self.roi_dataframe = roi_dataframe
        self.preferences = preferences
        self.cell_types = cell_types

        # Calculate the number of ROI and Random regions needed based on batch size and random ratio desired
        self.number_roi = round(batch_size * (1. - random_ratio))
        self.number_random_regions = round(batch_size - self.number_roi)

        self.chromosome_sizes_dictionary = build_chrom_sizes_dict(chromosomes, chromosome_sizes)

        # Get the ROIPool and/or RandomRegionsPool
        self.ROI_pool = self.__get_ROIPool()

        self.RandomRegions_pool = self.__get_RandomRegionsPool()

    def __get_ROIPool(self):
        """
        Passes the attributes to the ROIPool class to build a pool of regions of interest

        :return: Initializes the object used to generate batches of peak centered training examples.
        """
        return pd.read_csv(self.roi_dataframe, sep="\t", header=0)

    def __get_RandomRegionsPool(self):
        """
        Passes the attributes to the RandomRegionsPool class to build a pool of randomly generated training examples

        :return: Initializes the object used to generate batches of randomly generated examples
        """
        return RandomRegionsPool(chromosome_sizes_dictionary=self.chromosome_sizes_dictionary,
                                 region_length=self.region_length,
                                 preferences=self.preferences,
                                 chromosomes=self.chromosomes,
                                 cell_types=self.cell_types)

    def __get_roi_regions_list(self, number_random_regions):
        """
        Create batches of examples with the size of number_random_regions

        :param number_random_regions: Number of random regions to generate per batch

        :return: Training examples generated from random regions of the genome
        """
        random_regions_list = []

        for idx in range(number_random_regions):
            random_regions_list.append(self.ROI_pool.sample(self.number_roi).to_numpy().tolist())

        return random_regions_list

    def __mix_regions(self):
        """
        A generator that will combine batches of examples from two different sources with the defined proportions

        :return: Yields batches of training examples.
        """
        # Initialize the random and ROI generators with the specified batch size based on the total batch size
        regions_list = self.RandomRegions_pool.get_regions_list(
            number_random_regions=self.number_random_regions)

        regions_list.extend(self.ROI_pool.sample(self.number_roi).values.tolist())

        return regions_list

    def __get_random_cell_data(self):
        """
        Get the cell line data from a random cell line from the meta table

        :return: returns the signal file and the binding data file from a random cell line
        """
        meta_row = self.meta_dataframe[self.meta_dataframe['Cell_Type'] == random.choice(self.cell_types)].reset_index(
            drop=True)

        return meta_row.loc[0, 'ATAC_Signal_File'], meta_row.loc[0, 'Binding_File']

    def __get_specific_cell_data(self, cell_type):
        """
        Get the cell line data from a random cell line from the meta table

        :return: returns the signal file and the binding data file from a random cell line
        """
        meta_row = self.meta_dataframe[self.meta_dataframe['Cell_Type'] == cell_type].reset_index(
            drop=True)

        return meta_row.loc[0, 'ATAC_Signal_File'], meta_row.loc[0, 'Binding_File']

    def batch_generator(self, random_cell_type=""):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :return: A batch of training examples centered on regions of interest
        """
        while True:
            inputs_batch, targets_batch = [], []

            for region in self.__mix_regions():
                if random_cell_type:
                    signal, binding = self.__get_random_cell_data()
                else:
                    signal, binding = self.__get_specific_cell_data(region[4])

                with load_2bit(self.sequence) as sequence_stream, load_bigwig(signal) as signal_stream, \
                        load_bigwig(binding) as binding_stream:
                    inputs_batch.append(get_input_matrix(rows=self.input_channels,
                                                         cols=self.region_length,
                                                         bp_order=["A", "C", "G", "T"],
                                                         signal_stream=signal_stream,
                                                         sequence_stream=sequence_stream,
                                                         chromosome=region[0],
                                                         start=region[1],
                                                         end=region[2],
                                                         scale_signal=self.scale_signal
                                                         )
                                        )

                    targets_batch.append(get_target_matrix(binding_stream,
                                                           chromosome=region[0],
                                                           start=region[1],
                                                           end=region[2],
                                                           bp_resolution=self.bp_resolution
                                                           )
                                         )

            yield np.array(inputs_batch), np.array(targets_batch)


class ValidationDataGenerator(keras.utils.Sequence):
    """
    Generate batches of examples for the chromosomes of interest

    This generator will create individual batches of examples from a pool of regions of interest and/or a pool of
    random regions from the genome. This class object helps keep track of all of the required inputs and how they are
    processed.

    This generator expects a meta_table with the header:

    TF | Cell Type | ATAC signal | ChIP signal | ATAC peaks | ChIP peaks
    """

    def __init__(self,
                 sequence,
                 bp_resolution,
                 input_channels,
                 scale_signal,
                 roi_dataframe,
                 meta_dataframe,
                 region_length,
                 batch_size
                 ):
        self.input_channels = input_channels
        self.bp_resolution = bp_resolution
        self.sequence = sequence
        self.scale_signal = scale_signal
        self.meta_dataframe = meta_dataframe
        self.region_length = region_length
        self.batch_size = batch_size
        self.ROI_pool = pd.read_csv(roi_dataframe, sep="\t", header=0)
        self.indexes = np.arange(self.ROI_pool.shape[0])

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.ROI_pool.shape[0] / self.batch_size))

    def __get_specific_cell_data(self, cell_type):
        """
        Get the cell line data from a random cell line from the meta table

        :return: returns the signal file and the binding data file from a random cell line
        """
        meta_row = self.meta_dataframe[self.meta_dataframe['Cell_Type'] == cell_type].reset_index(
            drop=True)

        return meta_row.loc[0, 'ATAC_Signal_File'], meta_row.loc[0, 'Binding_File']

    def __getitem__(self, index):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :return: A batch of training examples centered on regions of interest
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_ROI_pool_dataframe = self.ROI_pool.iloc[indexes,]

        inputs_batch, targets_batch = [], []

        for region in batch_ROI_pool_dataframe.to_numpy().tolist():
            signal, binding = self.__get_specific_cell_data(region[4])

            with load_2bit(self.sequence) as sequence_stream, load_bigwig(signal) as signal_stream, \
                    load_bigwig(binding) as binding_stream:
                inputs_batch.append(get_input_matrix(rows=self.input_channels,
                                                     cols=self.region_length,
                                                     bp_order=["A", "C", "G", "T"],
                                                     signal_stream=signal_stream,
                                                     sequence_stream=sequence_stream,
                                                     chromosome=region[0],
                                                     start=region[1],
                                                     end=region[2],
                                                     scale_signal=self.scale_signal
                                                     )
                                    )

                targets_batch.append(get_target_matrix(binding_stream,
                                                       chromosome=region[0],
                                                       start=region[1],
                                                       end=region[2],
                                                       bp_resolution=self.bp_resolution
                                                       )
                                     )

        return np.array(inputs_batch), np.array(targets_batch)


class MaxATACModel(object):
    """
    This object will organize the input model parameters and fit a maxATAC model.
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
        Initialize the maxATAC model with the input parameters and architecture

        :param arch: Neural network architecture to use: DCNN, resNet, UNet, multi-modal
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
        self.meta_path = meta_path

        # Set the random seed for the model
        random.seed(seed)

        # Import meta txt as dataframe
        self.meta_dataframe = import_meta(self.meta_path)

        # Find the unique number of cell types in the meta file
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

    def __get_cell_types(self):
        """
        Get the unique cell lines in the meta dataframe. This function requires that the cell types are in a column of
        a pandas dataframe named "Cell_Type"

        :return: A list of unique cell types
        """
        return self.meta_dataframe["Cell_Type"].unique().tolist()

    def __get_peak_paths(self):
        """
        Get the ATAC-seq and ChIP-seq peak paths from the meta data file and concatenate them

        :return: A list of unique paths to all ATAC-seq and ChIP-seq data
        """
        return self.meta_dataframe["ATAC_Peaks"].unique().tolist(), self.meta_dataframe["CHIP_Peaks"].unique().tolist()

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


def plot_metrics(training_history,
                 results_location,
                 metric,
                 style="ggplot",
                 val_metric="",
                 title="",
                 y_label="",
                 suffix=""
                 ):
    """
        Plot the loss, accuracy, or dice coefficient training history.

        :param results_location:
        :param training_history:
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

    elif metric == 'binary_accuracy':
        val_metric = "val_binary_accuracy"
        title = "Model Accuracy"
        y_label = "Accuracy"
        suffix = "_model_accuracy"

    elif metric == "loss":
        val_metric = "val_loss"
        title = "Model Loss"
        y_label = "Loss"
        suffix = "_model_loss"

    elif metric == 'acc':
        val_metric = "val_acc"
        title = "Model Accuracy"
        y_label = "Accuracy"
        suffix = "_model_accuracy"

    else:
        pass

    # Get the training X and Y values
    t_y = training_history.history[metric]
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    # Get the validation X and Y values
    v_y = training_history.history[val_metric]
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
            remove_tags(results_location, "_{epoch}"),
            suffix + ".pdf"
        ),
        bbox_inches="tight"
    )

    # Close figure
    plt.close("all")
