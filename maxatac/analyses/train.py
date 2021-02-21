import logging

from maxatac.utilities.constants import TRAIN_MONITOR
from maxatac.utilities.system_tools import Mute

with Mute():  # hide stdout from loading the modules
    from maxatac.utilities.model_tools import get_callbacks
    from maxatac.utilities.training_tools import get_roi_pool, DataGenerator, MaxATACModel
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
    3) Read in the meta table
    4) Read in training and validation pool
    5) Initialize the training generator
    6) Initialize the validation generator
    7) Fit the models with the specific parameters

    :params args: seed, output, prefix, output_activation, lrate, decay, weights, quant, target_scale_factor, dense,
    batch_size, val_batch_size, train roi, validate roi, meta_file, sequence, average, threads, epochs, batches

    :returns: Trained models saved after each epoch
    """
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

    # Initialize the training generator
    train_gen = DataGenerator(sequence=args.sequence,
                              average=args.average,
                              meta_table=maxatac_model.meta_dataframe,
                              roi_pool=get_roi_pool(filepath=args.train_roi,
                                                    chroms=args.tchroms,
                                                    shuffle=True
                                                    ),
                              cell_type_list=maxatac_model.cell_types,
                              rand_ratio=args.rand_ratio,
                              chroms=args.tchroms,
                              quant=args.quant,
                              batch_size=args.batch_size,
                              target_scale_factor=args.target_scale_factor,
                              shuffle_cell_type=args.shuffle_cell_type
                              )

    # Initialize the validation generator
    val_gen = DataGenerator(sequence=args.sequence,
                            average=args.average,
                            meta_table=maxatac_model.meta_dataframe,
                            roi_pool=get_roi_pool(filepath=args.validate_roi,
                                                  chroms=args.vchroms,
                                                  shuffle=True
                                                  ),
                            cell_type_list=maxatac_model.cell_types,
                            rand_ratio=args.rand_ratio,
                            chroms=args.vchroms,
                            quant=args.quant,
                            batch_size=args.batch_size,
                            target_scale_factor=args.target_scale_factor,
                            shuffle_cell_type=args.shuffle_cell_type
                            )

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
        quant = args.quants
        tf = args.train_tf
        TCL = '_'.join(maxatac_model.cell_types)
        ARC = args.arch
        RR = args.rand_ratio
        export_model_structure(maxatac_model.nn_model, maxatac_model.results_location)

        if not quant:
            export_loss_dice_accuracy(training_history, tf, TCL, RR, ARC, maxatac_model.results_location)
        else:
            export_loss_mse_coeff(training_history, tf, TCL, RR, ARC, maxatac_model.results_location)

    logging.error("Results are saved to: " + maxatac_model.results_location)

# TODO write code to output model training statistics. Time to run and resources would be nice
