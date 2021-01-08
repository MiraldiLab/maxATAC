import logging

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

    # Initialize the model
    maxatac_model = GetModel(arch="DCNN_V2",
                             seed=args.seed,
                             OutDir=args.output,
                             prefix=args.prefix,
                             FilterNumber=args.FilterNumber,
                             KernelSize=args.KernelSize,
                             FilterScalingFactor=args.FilterScalingFactor,
                             threads=args.threads)

    # Initialize the training generator
    train_data_generator = DataGenerator(sequence=args.sequence,
                                         average=args.average,
                                         meta_table=args.meta_file,
                                         rand_ratio=args.trand_ratio,
                                         chroms=args.tchroms,
                                         batch_size=args.train_batch_size,
                                         blacklist=args.blacklist,
                                         chrom_sizes=args.chrom_sizes,
                                         chrom_pool_size=CHR_POOL_SIZE,
                                         bp_resolution=BP_RESOLUTION,
                                         region_length=INPUT_LENGTH,
                                         input_channels=INPUT_CHANNELS)

    # Initialize the validation data generator
    validate_data_generator = DataGenerator(sequence=args.sequence,
                                            average=args.average,
                                            meta_table=args.meta_file,
                                            rand_ratio=args.vrand_ratio,
                                            chroms=args.vchroms,
                                            batch_size=args.validate_batch_size,
                                            blacklist=args.blacklist,
                                            chrom_sizes=args.chrom_sizes,
                                            chrom_pool_size=CHR_POOL_SIZE,
                                            bp_resolution=BP_RESOLUTION,
                                            region_length=INPUT_LENGTH,
                                            input_channels=INPUT_CHANNELS)

    # Fit the model 
    maxatac_model.FitModel(train_gen=train_data_generator,
                           val_gen=validate_data_generator,
                           epochs=args.epochs,
                           train_batches=args.train_steps_per_epoch,
                           validate_batches=args.validate_steps_per_epoch)

    if args.plot:
        maxatac_model.PlotResults()

    logging.error("Results are saved to: " + maxatac_model.results_location)
