import logging
import sys
import random
from os import path
from maxatac.utilities.session import configure_session

from maxatac.utilities.system_tools import (
    get_dir,
    replace_extension,
    remove_tags,
    Mute
)

from maxatac.utilities.constants import TRAIN_MONITOR

from maxatac.utilities.plot import (
    export_model_loss,
    export_model_accuracy,
    export_model_dice,
    export_model_structure
)

from maxatac.utilities.genome_tools import DataGenerator

with Mute():  # hide stdout from loading the modules
    from maxatac.utilities.dcnn import (get_dilated_cnn, get_callbacks)


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

    def FitModel(self, train_gen, val_gen, batches, epochs):
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
            steps_per_epoch=batches,
            validation_steps=batches,
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


def run_training(args):
    """
    Train a maxATAC model

    Args
    ----
        args (obj):
            The argument parser object with the parameters from the parser

    Outputs
    -------
    """
    configure_session(1)

    # Initialize the model
    maxatac_model = GetModel(arch="DCNN_V2",
                             seed=args.seed,
                             OutDir=args.output,
                             prefix=args.prefix,
                             FilterNumber=args.FilterNumber,
                             KernelSize=args.KernelSize,
                             FilterScalingFactor=args.FilterScalingFactor,
                             threads=args.threads)

    train_data_generator = DataGenerator(sequence=args.sequence,
                                         average=args.average,
                                         meta_table=args.meta_file,
                                         rand_ratio=args.trand_ratio,
                                         chroms=args.tchroms,
                                         batch_size=args.batch_size,
                                         blacklist=args.blacklist,
                                         chrom_sizes=args.chrom_sizes)

    validate_data_generator = DataGenerator(sequence=args.sequence,
                                            average=args.average,
                                            meta_table=args.meta_file,
                                            rand_ratio=args.vrand_ratio,
                                            chroms=args.vchroms,
                                            batch_size=args.batch_size,
                                            blacklist=args.blacklist,
                                            chrom_sizes=args.chrom_sizes)

    # Fit the model 
    maxatac_model.FitModel(train_gen=train_data_generator,
                           val_gen=validate_data_generator,
                           batches=args.batches,
                           epochs=args.epochs)

    if args.plot:
        maxatac_model.PlotResults()

    logging.error("Results are saved to: " + maxatac_model.results_location)
