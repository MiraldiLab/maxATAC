import random
import numpy as np
import pandas as pd
import logging
import os
import sys

from os import path
from yaml import dump, safe_load
from datetime import datetime
from maxatac.utilities.session import configure_session
from maxatac.utilities.utils import TrainModel
from maxatac.utilities.helpers import (
    get_dir,
    replace_extension,
    remove_tags, 
    Mute
)

from maxatac.utilities.constants import (
    BP_RESOLUTION,
    TRAIN_MONITOR,
    INPUT_LENGTH
)

from maxatac.utilities.prepare import (
    get_roi_pool,
    pc_train_generator,
    create_val_generator
)

from maxatac.utilities.plot import (
    export_model_loss,
    export_model_accuracy,
    export_model_dice,
    export_model_structure
)

with Mute():  # hide stdout from loading the modules
    from maxatac.utilities.dcnn import (get_dilated_cnn, tp, tn, fp, fn, acc, get_callbacks)

def run_training(args):
    # TODO Random() object might be the same for all sub processes.
    random.seed(args.seed)
    
    #filenames to be decided...may need to change the below
    out_dir = get_dir(args.output)

    # Set up the results filename based on the epoch number
    results_filename = args.prefix + "_{epoch}" + ".h5"
    
    # This will create the results location and the file
    results_location = path.join(out_dir, results_filename)
    
    log_location = replace_extension(remove_tags(results_location, "_{epoch}"), ".csv")
    
    tensor_board_log_dir = get_dir(path.join(out_dir, "tensorboard"))
    
    configure_session(1)  # fit_generator should handle threads by itself
    
    if args.arch == "DCNN_V2":
        nn_model = get_dilated_cnn( input_filters=args.FILTER_NUMBER,
                                    input_kernel_size=args.KERNEL_SIZE,
                                    adam_learning_rate=args.lrate,
                                    adam_decay=args.decay,
                                    filters_scaling_factor=args.FILTERS_SCALING_FACTOR                                 
                                  )

    else:
        sys.exit("Model Architecture not specified correctly. Please check")
    
    meta_table = pd.read_csv(args.meta_file, sep='\t', header=0, index_col=None)

    train_pool = get_roi_pool(
        seq_len=INPUT_LENGTH,
        roi=args.train_roi,
        shuffle=True
    )
    
    validate_pool = get_roi_pool(
        seq_len=INPUT_LENGTH,
        roi=args.validate_roi,
        shuffle=False
    )
    
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
