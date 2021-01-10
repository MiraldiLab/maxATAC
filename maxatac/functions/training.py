import logging
import numpy as np

from maxatac.utilities.session import configure_session

from maxatac.utilities.constants import (BP_RESOLUTION,
                                         CHR_POOL_SIZE,
                                         INPUT_LENGTH,
                                         INPUT_CHANNELS)

from maxatac.utilities.genome_tools import DataGenerator

from maxatac.utilities.model_tools import GetModel

from keras.utils import Sequence


def run_training(args):
    """
    Train a maxATAC model

    :param args: The argument parser object with the parameters from the parser
    :return: a trained maxATAC model
    """
    # We configure the session to have the generator handle the multi-processing
    configure_session(1)

    logging.error("Loading model with parameters: \n"
                  "Seed: " + str(args.seed) + "\n" +
                  "Output Directory: " + args.output + "\n" +
                  "Filename Prefix: " + args.prefix + "\n" +
                  "Number of Filters: " + str(args.FilterNumber) + "\n" +
                  "Kernel Size in BP: " + str(args.KernelSize) + "\n" +
                  "FilterScalingFactor: " + str(args.FilterScalingFactor) + "\n" +
                  "Number of threads to use: " + str(args.threads))

    # Initialize the model
    maxatac_model = GetModel(arch="DCNN_V2",
                             seed=args.seed,
                             OutDir=args.output,
                             prefix=args.prefix,
                             FilterNumber=args.FilterNumber,
                             KernelSize=args.KernelSize,
                             FilterScalingFactor=args.FilterScalingFactor,
                             threads=args.threads)

    logging.error("Initializing the training generator with the parameters: " + "\n" +
                  "Training random ratio proportion: " + str(args.trand_ratio) + "\n" +
                  "Training chroms: " + str(args.tchroms) + "\n" +
                  "Training batch size: " + str(args.train_batch_size))

    # Initialize the training generator
    train_data_generator = DataGenerator(sequence=args.sequence,
                                         average=args.average,
                                         meta_table=args.meta_file,
                                         rand_ratio=args.train_rand_ratio,
                                         chroms=args.train_chroms,
                                         batch_size=args.train_batch_size,
                                         blacklist=args.blacklist,
                                         chrom_sizes=args.chrom_sizes,
                                         chrom_pool_size=CHR_POOL_SIZE,
                                         bp_resolution=BP_RESOLUTION,
                                         region_length=INPUT_LENGTH,
                                         input_channels=INPUT_CHANNELS,
                                         )

    logging.error("Initializing the validation generator with the parameters: " + "\n" +
                  "Validation random ratio proportion: " + str(args.vrand_ratio) + "\n" +
                  "Validation chroms: " + str(args.vchroms) + "\n" +
                  "Validation batch size: " + str(args.train_batch_size))

    # Initialize the validation data generator
    validate_data_generator = DataGenerator(sequence=args.sequence,
                                            average=args.average,
                                            meta_table=args.meta_file,
                                            rand_ratio=args.validate_rand_ratio,
                                            chroms=args.validate_chroms,
                                            batch_size=args.validate_batch_size,
                                            blacklist=args.blacklist,
                                            chrom_sizes=args.chrom_sizes,
                                            chrom_pool_size=CHR_POOL_SIZE,
                                            bp_resolution=BP_RESOLUTION,
                                            region_length=INPUT_LENGTH,
                                            input_channels=INPUT_CHANNELS)

    # Fit the model 
    maxatac_model.FitModel(train_gen=train_data_generator.BatchGenerator(),
                           val_gen=validate_data_generator.BatchGenerator(),
                           epochs=args.epochs,
                           train_batches=args.train_steps_per_epoch,
                           validation_batches=args.validate_steps_per_epoch)

    if args.plot:
        maxatac_model.PlotResults()

    logging.error("Results are saved to: " + maxatac_model.results_location)
