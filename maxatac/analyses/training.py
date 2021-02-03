import random
import pandas as pd
import logging
from os import path
from datetime import datetime
import sys
from maxatac.utilities.helpers import (
    get_dir,
    replace_extension,
    remove_tags
)
from maxatac.utilities.constants import (
    INPUT_CHANNELS,
    DNA_INPUT_CHANNELS,
    ATAC_INPUT_CHANNELS,
    INPUT_LENGTH,
    OUTPUT_FILTERS,
    PURE_CONV_LAYERS,
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
    KERNEL_INITIALIZER,
    BINARY_OUTPUT_ACTIVATION,
    QUANT_OUTPUT_ACTIVATION,
    PADDING,
    ADAM_BETA_1,
    ADAM_BETA_2,
    TRAIN_MONITOR
)

from maxatac.utilities.pc_prepare import (
    get_roi_pool,
    train_generator,
    create_val_generator
)

from maxatac.utilities.mute import Mute
from maxatac.utilities.session import configure_session
from maxatac.utilities.plot import (
    export_loss_dice_accuracy,
    export_loss_mse_coeff,
    export_model_structure
)

with Mute():  # hide stdout from loading the modules
    from maxatac.utilities.unet import get_callbacks
    from maxatac.utilities.d_cnn import get_dilated_cnn
    from maxatac.utilities.res_dcnn import get_res_dcnn
    from maxatac.utilities.multi_modal_models import MM_DCNN_V2


def run_training(args):
    # TODO Random() object might be the same for all sub processes.
    random.seed(args.seed)

    # Create a Results directory with Time Stamps
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = path.join(args.output, 'Run_' + ts)

    # filenames to be decided...may need to change the below
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

    configure_session(1)  # fit_generator should handle threads by itself

    if args.quant:
        output_activation = QUANT_OUTPUT_ACTIVATION
    else:
        output_activation = BINARY_OUTPUT_ACTIVATION

    if args.arch == "DCNN_V2":
        nn_model = get_dilated_cnn(input_length=INPUT_LENGTH,
                                   input_channels=INPUT_CHANNELS,
                                   input_filters=INPUT_FILTERS,
                                   input_kernel_size=INPUT_KERNEL_SIZE,
                                   input_activation=INPUT_ACTIVATION,
                                   output_filters=OUTPUT_FILTERS,
                                   output_kernel_size=OUTPUT_KERNEL_SIZE,
                                   output_activation=output_activation,
                                   filters_scaling_factor=FILTERS_SCALING_FACTOR,
                                   dilation_rate=DILATION_RATE,
                                   conv_blocks=CONV_BLOCKS,
                                   padding=PADDING,
                                   pool_size=POOL_SIZE,
                                   adam_learning_rate=args.lrate,
                                   adam_beta_1=ADAM_BETA_1,
                                   adam_beta_2=ADAM_BETA_2,
                                   adam_decay=args.decay,
                                   weights=args.weights,
                                   quant=args.quant
                                   )

    elif args.arch == "RES_DCNN_V2":
        nn_model = get_res_dcnn(input_length=INPUT_LENGTH,
                                input_channels=INPUT_CHANNELS,
                                input_filters=INPUT_FILTERS,
                                input_kernel_size=INPUT_KERNEL_SIZE,
                                input_activation=INPUT_ACTIVATION,
                                output_filters=OUTPUT_FILTERS,
                                output_kernel_size=OUTPUT_KERNEL_SIZE,
                                output_activation=output_activation,
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
                                weights=args.weights,
                                quant=args.quant
                                )

    elif args.arch == "MM_DCNN_V2":
        nn_model = MM_DCNN_V2(input_length=INPUT_LENGTH,
                              input_channels=INPUT_CHANNELS,
                              dna_input_channels=DNA_INPUT_CHANNELS,
                              atac_input_channels=ATAC_INPUT_CHANNELS,
                              input_filters=INPUT_FILTERS,
                              input_kernel_size=INPUT_KERNEL_SIZE,
                              input_activation=INPUT_ACTIVATION,
                              kernel_initializer=KERNEL_INITIALIZER,
                              output_filters=OUTPUT_FILTERS,
                              output_kernel_size=OUTPUT_KERNEL_SIZE,
                              output_activation=output_activation,
                              filters_scaling_factor=FILTERS_SCALING_FACTOR,
                              dilation_rate=DILATION_RATE,
                              output_length=OUTPUT_LENGTH,
                              pure_conv_layers=PURE_CONV_LAYERS,
                              conv_blocks=CONV_BLOCKS,
                              padding=PADDING,
                              pool_size=POOL_SIZE,
                              adam_learning_rate=args.lrate,
                              adam_beta_1=ADAM_BETA_1,
                              adam_beta_2=ADAM_BETA_2,
                              adam_decay=args.decay,
                              weights=args.weights,
                              quant=args.quant,
                              res_conn=False
                              )

    elif args.arch == "MM_Res_DCNN_V2":
        nn_model = MM_DCNN_V2(input_length=INPUT_LENGTH,
                              input_channels=INPUT_CHANNELS,
                              dna_input_channels=DNA_INPUT_CHANNELS,
                              atac_input_channels=ATAC_INPUT_CHANNELS,
                              input_filters=INPUT_FILTERS,
                              input_kernel_size=INPUT_KERNEL_SIZE,
                              input_activation=INPUT_ACTIVATION,
                              kernel_initializer=KERNEL_INITIALIZER,
                              output_filters=OUTPUT_FILTERS,
                              output_kernel_size=OUTPUT_KERNEL_SIZE,
                              output_activation=output_activation,
                              filters_scaling_factor=FILTERS_SCALING_FACTOR,
                              dilation_rate=DILATION_RATE,
                              output_length=OUTPUT_LENGTH,
                              pure_conv_layers=PURE_CONV_LAYERS,
                              conv_blocks=CONV_BLOCKS,
                              padding=PADDING,
                              pool_size=POOL_SIZE,
                              adam_learning_rate=args.lrate,
                              adam_beta_1=ADAM_BETA_1,
                              adam_beta_2=ADAM_BETA_2,
                              adam_decay=args.decay,
                              weights=args.weights,
                              quant=args.quant,
                              res_conn=True
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

    train_df = pd.read_csv(args.train_roi, sep='\t')
    train_cell_lines_set = set(train_df['Cell_Line'].unique())
    train_cell_lines = list(train_cell_lines_set)

    train_gen = train_generator(args.sequence,
                                args.average,
                                meta_table,
                                train_pool,
                                train_cell_lines,
                                args.rand_ratio,
                                args.train_tf,
                                args.tchroms,
                                bp_resolution=BP_RESOLUTION,
                                quant=args.quant,
                                batch_size=args.batch_size
                                )

    val_gen = create_val_generator(args.sequence,
                                   args.average,
                                   meta_table,
                                   train_cell_lines,
                                   args.train_tf,
                                   validate_pool,
                                   bp_resolution=BP_RESOLUTION,
                                   quant=args.quant,
                                   filters=None,
                                   val_batch_size=args.val_batch_size
                                   )

    training_history = nn_model.fit_generator(generator=train_gen,
                                              validation_data=val_gen,
                                              steps_per_epoch=args.batches,
                                              validation_steps=args.batches,
                                              epochs=args.epochs,
                                              callbacks=get_callbacks(model_location=results_location,
                                                                      log_location=log_location,
                                                                      tensor_board_log_dir=tensor_board_log_dir,
                                                                      monitor=TRAIN_MONITOR
                                                                      ),
                                              use_multiprocessing=args.threads > 1,
                                              workers=args.threads,
                                              verbose=1
                                              )

    if args.plot:
        quant = args.quant
        tf = args.train_tf
        TCL = '_'.join(train_cell_lines)
        ARC = args.arch
        RR = args.rand_ratio
        export_model_structure(nn_model, results_location)

        if not quant:
            export_loss_dice_accuracy(training_history, tf, TCL, RR, ARC, results_location)
        else:
            export_loss_mse_coeff(training_history, tf, TCL, RR, ARC, results_location)
    logging.error("Results are saved to: " + results_location)

#   1. Write code for model selection
# a. Create generator/dataset
# b. Make predictions
# c. Create AUPR curves
