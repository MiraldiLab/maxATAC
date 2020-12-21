
######
'''Making Changes for DCNN'''

######

import random
import numpy as np
import pandas as pd
import logging
import os
from os import path
from yaml import dump, safe_load
from datetime import datetime
import sys
from maxatac.utilities.helpers import (
    get_dir,
    get_rootname,
    replace_extension,
    remove_tags
)
from maxatac.utilities.constants import (
    INPUT_CHANNELS,
    INPUT_LENGTH,
    BATCH_SIZE,
    CHR_POOL_SIZE,
    BP_ORDER,
    OUTPUT_FILTERS,
    CONV_BLOCKS,
    DILATION_RATE,
    OUTPUT_LENGTH,
    BP_RESOLUTION,
    INPUT_FILTERS,
    POOL_SIZE,
    FILTERS_SCALING_FACTOR,
    INPUT_KERNEL_SIZE,
    OUTPUT_KERNEL_SIZE,
    INPUT_ACTIVATION,
    OUTPUT_ACTIVATION,
    PADDING,
    ADAM_BETA_1,
    ADAM_BETA_2,
    TRAIN_MONITOR,
    TRAIN_SCALE_SIGNAL
)
from maxatac.utilities.bigwig import load_bigwig, safe_load_bigwig
from maxatac.utilities.twobit import load_2bit

from maxatac.utilities.pc_prepare import (
    get_roi_pool,
    pc_train_generator,
    create_val_generator
)

from maxatac.utilities.mute import Mute
from maxatac.utilities.session import configure_session
from maxatac.utilities.plot import (
    export_model_loss,
    export_model_accuracy,
    export_model_dice,
    export_model_structure
)

with Mute():  # hide stdout from loading the modules
    from maxatac.utilities.unet import (
        get_unet,
        get_callbacks
    )
    from maxatac.utilities.d_cnn import (
        get_dilated_cnn, tp, tn, fp, fn, acc

    )
    from maxatac.utilities.res_dcnn import (
        get_res_dcnn, tp, tn, fp, fn, acc

    )

def run_pc_training(args):

    # TODO Random() object might be the same for all sub processes.
    random.seed(args.seed)
    #Create a Results directory with Time Stamps
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = path.join(args.output, 'Run_'+ ts)
    
    #filenames to be decided...may need to change the below
    out_dir = get_dir(out_dir)

    results_filename = args.prefix + \
        "_" + "_{epoch}" + \
        ".h5"
    results_location = path.join(out_dir, results_filename)
    log_location = replace_extension(
        remove_tags(results_location, "_{epoch}"),
        ".csv"
    )
    tensor_board_log_dir = get_dir(path.join(out_dir, "tensorboard"))
    
##############################################################################
    # logging.error(
    #     "Training" +
    #     "\n  Training signal: " + args.signal +
    #     "\n  Training binding sites: " + args.tsites +
    #     "\n  Validation signal: " + args.validation +
    #     "\n  Validation binding sites: " + args.vsites +
    #     "\n  Average signal: " + args.average +
    #     "\n  Sequence data: " + args.sequence +
    #     "\n  Limit training regions selection to: " + str(args.preferences) +
    #     "\n  Filters training regions signal based on: " + str(args.filters) +
    #     "\n  All chromosomes: \n   - " + "\n   - ".join(
    #             str(k) + ":" + \
    #             str(v["region"][0])+ "-" + \
    #             str(v["region"][1])+ " (" + \
    #             str(v["length"])+ ")" for k, v in args.chroms.items()
    #         ) +
    #     "\n  Training chromosomes: \n   - " + "\n   - ".join(
    #             str(k) + ":" + \
    #             str(v["region"][0])+ "-" + \
    #             str(v["region"][1])+ " (" + \
    #             str(v["length"])+ ")" for k, v in train_chroms.items()
    #         ) +
    #     "\n  Validation chromosomes: \n   - " + "\n   - ".join(
    #             str(k) + ":" + \
    #             str(v["region"][0])+ "-" + \
    #             str(v["region"][1])+ " (" + \
    #             str(v["length"])+ ")" for k, v in validate_chroms.items()
    #         ) +
    #     "\n  Load weights from: " + str(args.weights) +
    #     "\n  Proportion of training chromosomes: " + str(args.proportion) +
    #     "\n  Training epochs count: " + str(args.epochs) +
    #     "\n  Training batches per epoch: " + str(args.batches) +
    #     "\n  Batch size: " + str(BATCH_SIZE) +
    #     "\n  Learning rate: " + str(args.lrate) +
    #     "\n  Learning rate decay: " + str(args.decay) +
    #     "\n  Plot model structure and training history: " + str(args.plot) +
    #     "\n  Random seed: " + str(args.seed) +
    #     "\n  Threads count: " + str(args.threads) +
    #     "\n  Logging level: " + logging.getLevelName(args.loglevel) +
    #     "\n  Training log location: " +  log_location +
    #     "\n  TensorBoard logging directory: " +  tensor_board_log_dir +
    #     "\n  Results location: " + results_location
    # )
##############################################################################
    configure_session(1)  # fit_generator should handle threads by itself
    if args.arch == "DCNN_V2":
        nn_model = get_dilated_cnn( input_length=INPUT_LENGTH,
                                    input_channels=INPUT_CHANNELS,
                                    input_filters=INPUT_FILTERS,
                                    input_kernel_size=INPUT_KERNEL_SIZE,
                                    input_activation=INPUT_ACTIVATION,
                                    output_filters=OUTPUT_FILTERS,
                                    output_kernel_size=OUTPUT_KERNEL_SIZE,
                                    output_activation=OUTPUT_ACTIVATION,
                                    filters_scaling_factor=FILTERS_SCALING_FACTOR,
                                    dilation_rate=DILATION_RATE,
                                    conv_blocks=CONV_BLOCKS,
                                    padding=PADDING,
                                    pool_size=POOL_SIZE,
                                    adam_learning_rate=args.lrate,
                                    adam_beta_1=ADAM_BETA_1,
                                    adam_beta_2=ADAM_BETA_2,
                                    adam_decay=args.decay,
                                    weights=args.weights
                                  )
    elif args.arch == "RES_DCNN_V2":
        nn_model = get_res_dcnn( input_length=INPUT_LENGTH,
                                 input_channels=INPUT_CHANNELS,
                                 input_filters=INPUT_FILTERS,
                                 input_kernel_size=INPUT_KERNEL_SIZE,
                                 input_activation=INPUT_ACTIVATION,
                                 output_filters=OUTPUT_FILTERS,
                                 output_kernel_size=OUTPUT_KERNEL_SIZE,
                                 output_activation=OUTPUT_ACTIVATION,
                                 filters_scaling_factor=FILTERS_SCALING_FACTOR,
                                 dilation_rate=DILATION_RATE,
                                 output_length=OUTPUT_LENGTH,
                                 conv_blocks=CONV_BLOCKS,
                                 padding=PADDING,
                                 pool_size=POOL_SIZE,
                                 adam_learning_rate=args.lrate,
                                 adam_beta_1=ADAM_BETA_1,
                                 adam_beta_2=ADAM_BETA_2,
                                 adam_decay=args.decay,
                                 weights=args.weights
                                 )
    else:
        sys.exit("Model Architecture not specified correctly. Please check")
    meta_table = pd.read_csv(args.meta_file, sep='\t', header=0, index_col=None)

    train_pool = get_roi_pool(
        seq_len=INPUT_LENGTH,
        roi=args.train_roi,
        shuffle=True
    )
    

#Commented out as this Validate Pool is used to generate validation regions associated with PCPC regions specified in validate_roi. If we want to do this use this validate_pool
    validate_pool = get_roi_pool(
        seq_len=INPUT_LENGTH,
        roi=args.validate_roi,
        shuffle=False
    )
    
    #test_cell_lines = set([str(item) for item in args.test_cell_lies.split(',')])
    test_cell_lines = set(args.test_cell_lines)
    all_cell_lines = set(meta_table["Cell_Line"].unique())
    train_cell_lines = list(all_cell_lines - test_cell_lines)

    train_gen = pc_train_generator( args.sequence,
                                    args.average,
                                    meta_table,
                                    train_pool,
                                    train_cell_lines,
                                    args.rand_ratio,
                                    args.train_tf,
                                    args.tchroms,
                                    bp_resolution=BP_RESOLUTION,
                                    filters=None
                                 )
    
    val_gen = create_val_generator( args.sequence,
                                    args.average,
                                    meta_table,
                                    train_cell_lines,
                                    args.train_tf,
                                    validate_pool,
                                    bp_resolution=BP_RESOLUTION,
                                    filters=None
                                    )
    
    training_history = nn_model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        steps_per_epoch=args.batches,
        validation_steps=args.batches,
        epochs=args.epochs,
        callbacks=get_callbacks(
            model_location=results_location,
            log_location=log_location,
            tensor_board_log_dir=tensor_board_log_dir,
            monitor=TRAIN_MONITOR
        ),
        use_multiprocessing=args.threads > 1,
        workers=args.threads,
        verbose=1
    )
    
    if args.plot:
        export_model_structure(nn_model, results_location)
        export_model_loss(training_history, results_location)
        export_model_dice(training_history, results_location)
        export_model_accuracy(training_history, results_location)

    logging.error("Results are saved to: " + results_location)

#   1. Write code for model selection
        # a. Create generator/dataset
        # b. Make predictions
        # c. Create AUPR curves