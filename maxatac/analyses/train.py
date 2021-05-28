import logging
import sys
import timeit
import pandas as pd

from maxatac.utilities.constants import TRAIN_MONITOR
from maxatac.utilities.system_tools import Mute

with Mute():  # hide stdout from loading the modules
    from maxatac.utilities.model_tools import get_callbacks
    from maxatac.utilities.training_tools import DataGenerator, MaxATACModel, ROIPool
    from maxatac.utilities.plot import export_loss_dice_accuracy, export_loss_mse_coeff, export_model_structure


def run_training(args):
    """
    Train a maxATAC model using ATAC-seq and ChIP-seq data

    The primary input to the training function is a meta file that contains all of the information for the locations of
    ATAC-seq signal, ChIP-seq signal, TF, and Cell type.

    Example header for meta file. The meta file must be a tsv file, but the order of the columns does not matter. As
    long as the column names are the same:

    TF | Cell_Type | ATAC_Signal_File | Binding_File | ATAC_Peaks | ChIP_peaks

    _________________
    Workflow Overview

    1) Set up the directories and filenames
    2) Initialize the model based on the desired architectures
    3) Read in training and validation pools
    4) Initialize the training and validation generators
    5) Fit the models with the specific parameters

    :params args: seed, output, prefix, output_activation, lrate, decay, weights, quant, target_scale_factor, dense,
    batch_size, val_batch_size, train roi, validate roi, meta_file, sequence, average, threads, epochs, batches

    :returns: Trained models saved after each epoch
    """
    startTime = timeit.default_timer()
    
    # Initialize the model with the architecture of choice
    maxatac_model = MaxATACModel(arch=args.arch,
                                 seed=args.seed,
                                 output_directory=args.output,
                                 prefix=args.prefix,
                                 threads=args.threads,
                                 meta_path=args.meta_file,
                                 quant=args.quant,
                                 output_activation=args.output_activation,
                                 target_scale_factor=args.target_scale_factor,
                                 dense=args.dense,
                                 weights=args.weights
                                 )

    logging.error("Window the genome to desired bin width and step size")

    # Import windowed genome
    windowed_gen = pd.read_csv(args.window_sequence, sep='\t', header=None, names=["Chr", "Start", "Stop"])

    logging.error("Import training regions")

    # Import training regions
    train_examples = ROIPool(chroms=args.tchroms,
                             roi_file_path=args.roi,
                             prefix=args.prefix,
                             output_directory=maxatac_model.output_directory,
                             tag="training",
                             test_cell_type=maxatac_model.test_cell_type,
                             genomic_bins=windowed_gen)

    logging.error("Import validation regions")

    # Import validation regions
    validate_examples = ROIPool(chroms=args.vchroms,
                                roi_file_path=args.roi,
                                prefix=args.prefix,
                                output_directory=maxatac_model.output_directory,
                                tag="validation",
                                test_cell_type=maxatac_model.test_cell_type,
                                genomic_bins=windowed_gen)

    logging.error("Initialize the training generator")

    # Initialize the training generator
    train_gen = DataGenerator(sequence=args.sequence,
                              meta_table=maxatac_model.meta_dataframe,
                              roi_pool=train_examples,
                              cell_type_list=maxatac_model.cell_types,
                              chroms=args.tchroms,
                              quant=args.quant,
                              batch_size=args.batch_size,
                              target_scale_factor=args.target_scale_factor,
                              shuffle_cell_type=args.shuffle_cell_type
                              )

    logging.error("Initialize the validation generator")

    # Initialize the validation generator
    val_gen = DataGenerator(sequence=args.sequence,
                            meta_table=maxatac_model.meta_dataframe,
                            roi_pool=validate_examples,
                            cell_type_list=maxatac_model.cell_types,
                            chroms=args.vchroms,
                            quant=args.quant,
                            batch_size=args.batch_size,
                            target_scale_factor=args.target_scale_factor,
                            shuffle_cell_type=args.shuffle_cell_type
                            )

    logging.error("Fit the model")

    # Fit the model
    training_history = maxatac_model.nn_model.fit_generator(generator=train_gen,
                                                            validation_data=val_gen,
                                                            steps_per_epoch=args.batches,
                                                            validation_steps=args.batches,
                                                            epochs=args.epochs,
                                                            callbacks=get_callbacks(
                                                                model_location=maxatac_model.results_location,
                                                                log_location=maxatac_model.log_location,
                                                                tensor_board_log_dir=maxatac_model.tensor_board_log_dir,
                                                                monitor=TRAIN_MONITOR
                                                            ),
                                                            use_multiprocessing=args.threads > 1,
                                                            workers=args.threads,
                                                            verbose=1
                                                            )

    # If plot then plot the model structure and training metrics
    if args.plot:
        quant = args.quant
        tf = maxatac_model.train_tf
        TCL = '_'.join(maxatac_model.cell_types)
        ARC = args.arch
        RR = "NaN"
        export_model_structure(maxatac_model.nn_model, maxatac_model.results_location)

        if not quant:
            export_loss_dice_accuracy(training_history, tf, TCL, RR, ARC, maxatac_model.results_location)
        else:
            export_loss_mse_coeff(training_history, tf, TCL, RR, ARC, maxatac_model.results_location)

    logging.error("Results are saved to: " + maxatac_model.results_location)
    
    stopTime = timeit.default_timer()
    totalTime = stopTime - startTime
    
    # output running time in a nice format.
    mins, secs = divmod(totalTime, 60)
    hours, mins = divmod(mins, 60)

    logging.error("Total training time: %d:%d:%d.\n" % (hours, mins, secs))

    sys.exit()

