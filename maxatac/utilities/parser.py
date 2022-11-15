import argparse
import random
import os
from os import getcwd

from pkg_resources import require
from yaml import dump

from maxatac.utilities.system_tools import (get_version,
                                            get_absolute_path,
                                            get_cpu_count,
                                            Mute,
                                            update_reference_genome_paths
                                            )

with Mute():
    from maxatac.analyses.average import run_averaging
    from maxatac.analyses.predict import run_prediction
    from maxatac.analyses.train import run_training
    from maxatac.analyses.normalize import run_normalization
    from maxatac.analyses.benchmark import run_benchmarking
    from maxatac.analyses.peaks import run_call_peaks
    from maxatac.analyses.variants import run_variants
    from maxatac.analyses.prepare import run_prepare
    from maxatac.analyses.threshold import run_thresholding
    from maxatac.analyses.data import run_data

from maxatac.utilities.constants import (DEFAULT_TRAIN_VALIDATE_CHRS,
                                         LOG_LEVELS,
                                         DEFAULT_LOG_LEVEL,
                                         DEFAULT_TRAIN_EPOCHS,
                                         DEFAULT_TRAIN_BATCHES_PER_EPOCH,
                                         BATCH_SIZE,
                                         VAL_BATCH_SIZE,
                                         INPUT_LENGTH,
                                         DEFAULT_TRAIN_CHRS,
                                         DEFAULT_VALIDATE_CHRS,
                                         DEFAULT_ROUND,
                                         DEFAULT_TEST_CHRS,
                                         DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
                                         DEFAULT_BENCHMARKING_BIN_SIZE,
                                         ALL_CHRS,
                                         AUTOSOMAL_CHRS
                                         )


def normalize_args(args, skip_list=[], cwd_abs_path=None):
    """
    Converts all relative path arguments to absolute
    ones relatively to the cwd_abs_path or current working directory.
    Skipped arguments and None will be returned unchanged.
    """
    cwd_abs_path = getcwd() if cwd_abs_path is None else cwd_abs_path
    normalized_args = {}
    for key, value in args.__dict__.items():
        if key not in skip_list and value is not None:
            if isinstance(value, list):
                for v in value:
                    normalized_args.setdefault(key, []).append(
                        get_absolute_path(v, cwd_abs_path)
                    )
            else:
                normalized_args[key] = get_absolute_path(value, cwd_abs_path)
        else:
            normalized_args[key] = value
    return argparse.Namespace(**normalized_args)


def get_parser():
    """Build parsers with user input.
    
    There are currently parsers for the following subcommands:
    average, train, predict, normalize, data, variants, benchmark,
    threshold, peaks, prepare
    
    Returns:
        argparse object
    """
    # Parent (general) parser
    parent_parser = argparse.ArgumentParser(add_help=False)

    general_parser = argparse.ArgumentParser(description="Neural networks for predicting TF binding using ATAC-seq")

    # Add subparsers to the general parser and require that one is provided
    subparsers = general_parser.add_subparsers()
    subparsers.required = True

    general_parser.add_argument("--version",
                                action="version",
                                version=get_version(),
                                help="Print version information and exit"
                                )

    general_parser.add_argument("--genome",
                                dest="genome",
                                type=str,
                                default="hg38",
                                required=False,
                                help="The reference genome build to use."
                                )

    #############################################
    # Data subparser
    #############################################
    data_parser = subparsers.add_parser("data",
                                        parents=[parent_parser],
                                        help="Download and install reference data."
                                        )

    # Set the default function
    data_parser.set_defaults(func=run_data)

    # Add arguments to the parser
    data_parser.add_argument("--genome",
                             dest="genome",
                             type=str,
                             nargs="+",
                             default=["hg38"],
                             required=False,
                             help="The reference genome build to download."
                             )

    data_parser.add_argument("--output", "-o",
                             dest="output",
                             type=str,
                             default=os.path.join(os.path.expanduser('~'), "opt"),
                             required=False,
                             help="Output results directory."
                             )

    data_parser.add_argument("--loglevel",
                             dest="loglevel",
                             type=str,
                             default="info",
                             choices=LOG_LEVELS.keys(),
                             help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                             )

    #############################################
    # Average subparser
    #############################################
    # Add a subparser for the average function
    average_parser = subparsers.add_parser("average",
                                           parents=[parent_parser],
                                           help="Average bigwig files together."
                                           )

    # Set the default function
    average_parser.set_defaults(func=run_averaging)

    # Add required arguments to the parser
    average_parser.add_argument("-i",
                                dest="bigwig_files",
                                type=str,
                                nargs="+",
                                required=True,
                                help="Input bigwig files."
                                )

    average_parser.add_argument("-n", "--name", "--prefix",
                                dest="name",
                                type=str,
                                required=True,
                                help="Output filename base. The extension .bw will be added to the name."
                                )
    # Add optional arguments to the parser
    average_parser.add_argument("-cs", "--chrom_sizes", "--chromosome_sizes",
                                dest="chromosome_sizes",
                                type=str,
                                help="Input chromosome sizes file"
                                )

    average_parser.add_argument("-c", "--chroms", "--chromosomes",
                                dest="chromosomes",
                                type=str,
                                nargs="+",
                                default=AUTOSOMAL_CHRS,
                                help="Chromosomes for averaging. Default: 1-22"
                                )

    average_parser.add_argument("-o", "--output", "--output_dir",
                                dest="output_dir",
                                type=str,
                                default=os.getcwd(),
                                help="Output directory. Default: Output to current working directory."
                                )

    average_parser.add_argument("--loglevel",
                                dest="loglevel",
                                type=str,
                                default="info",
                                choices=LOG_LEVELS.keys(),
                                help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                )

    #############################################
    # Predict subparser
    #############################################
    predict_parser = subparsers.add_parser("predict",
                                           parents=[parent_parser],
                                           help="Predict transcription factor binding.",
                                           )

    # Set the default function
    predict_parser.set_defaults(func=run_prediction)

    # Add arguments to the parser
    group = predict_parser.add_mutually_exclusive_group(required=True)

    group.add_argument("-tf", "--tf_name",
                       dest="TF",
                       type=str,
                       help="The TF name for prediction"
                       )

    group.add_argument("-m", "--model",
                       dest="model",
                       type=str,
                       help="Trained maxATAC model .h5 file."
                       )

    predict_parser.add_argument("--seq", "--sequence",
                                dest="sequence",
                                type=str,
                                help="Genome sequence 2bit file."
                                )

    predict_parser.add_argument("-i", "-s", "--signal",
                                dest="signal",
                                type=str,
                                required=True,
                                help="Input ATACseq bigwig file."
                                )

    predict_parser.add_argument("-o", "--output",
                                dest="output_directory",
                                type=str,
                                default="./prediction_results",
                                help="Folder for prediction results. Default: ./prediction_results"
                                )

    predict_parser.add_argument("-bl", "--blacklist",
                                dest="blacklist",
                                type=str,
                                help="The blacklisted regions to exclude in BED format"
                                )

    predict_parser.add_argument("--bed", "--peaks", "--regions", "-roi",
                                dest="roi",
                                default=False,
                                required=False,
                                help="Bed file with ranges for input sequences to be used in prediction. \
                                      Default: None, predictions are done on the whole chromosome."
                                )

    predict_parser.add_argument("--windows", "-w",
                                dest="windows",
                                required=False,
                                help="Bed file with ranges to use for windows. These windows must 1,024 bp wide and \
                                    generated with an uniform step size. These will replace windows generated de novo."
                                )

    predict_parser.add_argument("--loglevel",
                                dest="loglevel",
                                type=str,
                                default="info",
                                choices=LOG_LEVELS.keys(),
                                help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                )

    predict_parser.add_argument("--batch_size",
                                dest="batch_size",
                                type=int,
                                default=10000,
                                help="Number of regions to predict on at a time"
                                )

    predict_parser.add_argument("--step_size",
                                dest="step_size",
                                type=int,
                                default=int(INPUT_LENGTH / 4),
                                help="Step size to use to build sliding window regions"
                                )

    predict_parser.add_argument("-n", "--name", "--prefix",
                                dest="name",
                                required=True,
                                type=str,
                                help="Sting to use for filename. This should not include extensions. \
                                      Example: GM12878_CTCF"
                                )

    predict_parser.add_argument("-cs", "--chrom_sizes", "--chrom_sizes",
                                dest="chrom_sizes",
                                type=str,
                                help="Chromosome sizes file"
                                )

    predict_parser.add_argument("-c", "-chroms", "--chromosomes",
                                dest="chromosomes",
                                type=str,
                                nargs="+",
                                default=AUTOSOMAL_CHRS,
                                help="Chromosomes from --chromosomes fixed for prediction. \
                                      Default: All chromosomes chr1-22"
                                )

    predict_parser.add_argument("-bin", "--bin_size",
                                dest="BIN_SIZE",
                                type=int,
                                default=DEFAULT_BENCHMARKING_BIN_SIZE,
                                help="Bin size to use for peak calling"
                                )

    predict_parser.add_argument("-cutoff_type", "--cutoff_type",
                                dest="cutoff_type",
                                default="F1",
                                type=str,
                                help="Cutoff type (i.e. Precision)"
                                )

    predict_parser.add_argument("-cutoff_value", "--cutoff_value",
                                dest="cutoff_value",
                                type=float,
                                help="Cutoff value for the cutoff type provided. Not used with F1 score."
                                )

    predict_parser.add_argument("-cutoff_file", "--cutoff_file",
                                dest="cutoff_file",
                                type=str,
                                help="Cutoff file provided in /data/models"
                                )

    predict_parser.add_argument("-skip_call_peaks", "--skip_call_peaks",
                                dest="skip_call_peaks",
                                action="store_true",
                                default=False,
                                required=False,
                                help="Skip calling peaks on prediction tracks"
                                )

    #############################################
    # Train parser
    #############################################
    train_parser = subparsers.add_parser("train",
                                         parents=[parent_parser],
                                         help="Train a maxATAC model."
                                         )

    # Set the default function
    train_parser.set_defaults(func=run_training)

    # Add arguments to the parser
    train_parser.add_argument("--genome",
                              dest="genome",
                              type=str,
                              default="hg38",
                              required=False,
                              help="The reference genome build to use."
                              )

    train_parser.add_argument("--sequence",
                              dest="sequence",
                              type=str,
                              help="Genome sequence 2bit file"
                              )

    train_parser.add_argument("--meta_file",
                              dest="meta_file",
                              type=str,
                              required=True,
                              help="Meta file containing ATAC Signal and peak path for all cell lines (.tsv format)"
                              )

    train_parser.add_argument("--train_roi",
                              dest="train_roi",
                              type=str,
                              required=False,
                              help="Optional BED format file that will be used as the training regions of interest "
                                   "instead of using the peak files to build training regions"
                              )

    train_parser.add_argument("--validate_roi",
                              dest="validate_roi",
                              type=str,
                              required=False,
                              help="Optional BED format file that will be used as the validation regions of interest "
                                   "instead of using the peak files to build validation regions"
                              )

    train_parser.add_argument("--output_activation",
                              dest="output_activation",
                              type=str,
                              required=False,
                              default="sigmoid",
                              help="Activation function used for model output layer. Default: sigmoid"
                              )

    train_parser.add_argument("--chroms",
                              dest="chroms",
                              type=str,
                              nargs="+",
                              required=False,
                              default=DEFAULT_TRAIN_VALIDATE_CHRS,
                              help="Chromosome list to use for training and validation."
                              )

    train_parser.add_argument("--tchroms",
                              dest="tchroms",
                              type=str,
                              nargs="+",
                              required=False,
                              default=DEFAULT_TRAIN_CHRS,
                              help="Chromosome list to use for training."
                              )

    train_parser.add_argument("--vchroms",
                              dest="vchroms",
                              type=str,
                              nargs="+",
                              required=False,
                              default=DEFAULT_VALIDATE_CHRS,
                              help="Chromosome list to use for validation"
                              )

    train_parser.add_argument("--arch",
                              dest="arch",
                              type=str,
                              required=False,
                              default="DCNN_V2",
                              help="Specify the model architecture. Currently support DCNN_V2, RES_DCNN_V2, "
                                   "MM_DCNN_V2 and MM_Res_DCNN_V2 "
                              )

    train_parser.add_argument("--rand_ratio",
                              dest="rand_ratio",
                              type=float,
                              required=False,
                              default=0,
                              help="Ratio for controlling fraction of random sequences in each training batch. "
                                   "Default: 0 "
                              )

    train_parser.add_argument("--seed",
                              dest="seed",
                              type=int,
                              default=random.randint(1, 99999),
                              help="Seed for pseudo-random generanor. Default: random int [1, 99999]"
                              )

    train_parser.add_argument("--weights",
                              dest="weights",
                              type=str,
                              help="Weights to initialize model before training. Default: do not load"
                              )

    train_parser.add_argument("--epochs",
                              dest="epochs",
                              type=int,
                              default=DEFAULT_TRAIN_EPOCHS,
                              help="Number of training epochs. Default: " + str(DEFAULT_TRAIN_EPOCHS)
                              )

    train_parser.add_argument("--batches",
                              dest="batches",
                              type=int,
                              default=DEFAULT_TRAIN_BATCHES_PER_EPOCH,
                              help="Number of training batches per epoch. Default: " + str(
                                  DEFAULT_TRAIN_BATCHES_PER_EPOCH)
                              )

    train_parser.add_argument("--batch_size",
                              dest="batch_size",
                              type=int,
                              default=BATCH_SIZE,
                              help="Number of examples per batch. Default: " + str(BATCH_SIZE)
                              )

    train_parser.add_argument("--val_batch_size",
                              dest="val_batch_size",
                              type=int,
                              default=VAL_BATCH_SIZE,
                              help="Number of examples per batch. Default: " + str(VAL_BATCH_SIZE)
                              )

    train_parser.add_argument("--prefix",
                              dest="prefix",
                              type=str,
                              default="maxatac_model",
                              help="Output prefix. Default: weights"
                              )

    train_parser.add_argument("--output",
                              dest="output",
                              type=str,
                              default="./training_results",
                              help="Folder for training results. Default: ./training_results"
                              )

    train_parser.add_argument("--plot",
                              dest="plot",
                              action="store_true",
                              default=True,
                              help="Plot model structure and training history. Default: True"
                              )

    train_parser.add_argument("--dense",
                              dest="dense",
                              action="store_true",
                              default=False,
                              help="If True, then make a dense layer before model output. Default: False"
                              )

    train_parser.add_argument("--threads",
                              dest="threads",
                              type=int,
                              default=get_cpu_count(),
                              help="Number of processes to run training in parallel. Default: 1"
                              )

    train_parser.add_argument("--loglevel",
                              dest="loglevel",
                              type=str,
                              default="info",
                              choices=LOG_LEVELS.keys(),
                              help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                              )

    train_parser.add_argument("--shuffle_cell_type",
                              dest="shuffle_cell_type",
                              action="store_true",
                              default=True,
                              help="If shuffle_cell_type, then shuffle training ROI cell type label"
                              )

    train_parser.add_argument("--rev_comp",
                              dest="rev_comp",
                              action="store_true",
                              default=False,
                              help="If rev_comp, then use the reverse complement in training"
                              )

    train_parser.add_argument("--multiprocessing",
                              dest="multiprocessing",
                              action="store_true",
                              default=False,
                              help="If multiprocessing, then use multiprocessing with tf.keras.fit()"
                              )

    train_parser.add_argument("--max_queue_size",
                              dest="max_queue_size",
                              help="The max number of workers to spin up. These workers will load data and wait for fit."
                              )

    train_parser.add_argument("--save_roi",
                              dest="save_roi",
                              action="store_true",
                              default=False,
                              help="Saves ROI file and stats generated for training"
                              )

    train_parser.add_argument("--blacklist",
                              dest="blacklist",
                              type=str,
                              help="Blacklist regions to exclude in BED format"
                              )

    train_parser.add_argument("--chrom_sizes",
                              dest="chrom_sizes",
                              type=str,
                              help="Chromosome sizes file"
                              )

    #############################################
    # Normalize parser
    #############################################
    # Add a subparser for the normalize function
    normalize_parser = subparsers.add_parser("normalize",
                                             parents=[parent_parser],
                                             help="Normalize bigwig signal tracks."
                                             )

    # Set the default function
    normalize_parser.set_defaults(func=run_normalization)

    # Add arguments to the parser
    normalize_parser.add_argument("-i", "--signal",
                                  dest="signal",
                                  type=str,
                                  required=True,
                                  help="Input .bigwig file."
                                  )

    normalize_parser.add_argument("-n", "--name", "--prefix",
                                  required=True,
                                  dest="name",
                                  type=str,
                                  help="Name to use for filename"
                                  )

    normalize_parser.add_argument("-cs", "--chrom_sizes", "--chromosome_sizes",
                                  dest="chrom_sizes",
                                  type=str,
                                  help="Chrom sizes file"
                                  )

    normalize_parser.add_argument("-c", "--chroms", "--chromosomes",
                                  dest="chromosomes",
                                  type=str,
                                  nargs="+",
                                  default=AUTOSOMAL_CHRS,
                                  help="Chromosomes for normalization. Default: 1-22"
                                  )

    normalize_parser.add_argument("-o", "--output", "--output_dir",
                                  dest="output_dir",
                                  type=str,
                                  default=os.getcwd(),
                                  help="Output directory. Default: Output to current working directory."
                                  )

    normalize_parser.add_argument("--min",
                                  dest="min",
                                  required=False,
                                  type=int,
                                  default=0,
                                  help="The minimum value to use for normalization"
                                  )

    normalize_parser.add_argument("--max",
                                  dest="max",
                                  type=int,
                                  required=False,
                                  default=False,
                                  help="The maximum value to use for normalization"
                                  )

    normalize_parser.add_argument("--clip",
                                  dest="clip",
                                  type=bool,
                                  required=False,
                                  default=False,
                                  help="Whether to clip minmax values to the range 0,1"
                                  )

    normalize_parser.add_argument("--method",
                                  dest="method",
                                  type=str,
                                  default="min-max",
                                  help="The method to use for normalization"
                                  )

    normalize_parser.add_argument("--max_percentile",
                                  dest="max_percentile",
                                  type=int,
                                  default=99,
                                  help="The maximum percentile to use for normalization"
                                  )

    normalize_parser.add_argument("--loglevel",
                                  dest="loglevel",
                                  type=str,
                                  default="info",
                                  choices=LOG_LEVELS.keys(),
                                  help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                  )

    normalize_parser.add_argument("--blacklist_bw",
                                  dest="blacklist_bw",
                                  type=str,
                                  help="The blacklisted regions to exclude (BigWig file format)"
                                  )

    #############################################
    # Benchmark subparser
    #############################################
    benchmark_parser = subparsers.add_parser("benchmark",
                                             parents=[parent_parser],
                                             help="Benchmark predictions against a gold standard."
                                             )

    # Set the default function
    benchmark_parser.set_defaults(func=run_benchmarking)

    # Add arguments to the parser
    # Add arguments to the parser
    benchmark_prediction_filetype = benchmark_parser.add_mutually_exclusive_group(required=True)

    benchmark_prediction_filetype.add_argument("-bed", "--bed",
                                               dest="prediction",
                                               type=str,
                                               help="The TF name for prediction"
                                               )

    benchmark_prediction_filetype.add_argument("--bw", "--bigwig", "-bw",
                                               dest="prediction",
                                               type=str,
                                               help="Prediction bigWig file"
                                               )

    benchmark_parser.add_argument("--gold_standard",
                                  dest="gold_standard",
                                  type=str,
                                  required=True,
                                  help="Gold Standard file"
                                  )

    benchmark_parser.add_argument("-c", "--chroms", "--chromosomes",
                                  dest="chromosomes",
                                  type=str,
                                  nargs="+",
                                  default=DEFAULT_TEST_CHRS,
                                  help="Chromosomes list for analysis. \
                                        Optionally with regions in a form of chrN:start-end. \
                                        Default: main human chromosomes, whole length"
                                  )

    benchmark_parser.add_argument("--bin_size",
                                  dest="bin_size",
                                  type=int,
                                  default=DEFAULT_BENCHMARKING_BIN_SIZE,
                                  help="Bin size to split prediction and control data before running prediction. \
                                        Default: " + str(DEFAULT_BENCHMARKING_BIN_SIZE)
                                  )

    benchmark_parser.add_argument("--agg",
                                  dest="agg_function",
                                  type=str,
                                  default=DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
                                  help="Aggregation function to use for combining results into bins: \
                                        max, mean, min"
                                  )

    benchmark_parser.add_argument("--round_predictions",
                                  dest="round_predictions",
                                  type=int,
                                  default=DEFAULT_ROUND,
                                  help="Round binned values to this number of decimal places"
                                  )

    benchmark_parser.add_argument("-n", "--name", "--prefix",
                                  dest="prefix",
                                  type=str,
                                  required=True,
                                  help="Prefix for the file name"
                                  )

    benchmark_parser.add_argument("-o", "--output_directory",
                                  dest="output_directory",
                                  type=str,
                                  default="./benchmarking_results",
                                  help="Folder for benchmarking results. Default: ./benchmarking_results"
                                  )

    benchmark_parser.add_argument("--loglevel",
                                  dest="loglevel",
                                  type=str,
                                  default="info",
                                  choices=LOG_LEVELS.keys(),
                                  help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                  )

    benchmark_parser.add_argument("--blacklist_bw",
                                  dest="blacklist_bw",
                                  type=str,
                                  help="The blacklisted regions to exclude in BigWig format"
                                  )

    benchmark_parser.add_argument("-skip_plot", "--skip_plot",
                                dest="skip_plot",
                                action="store_true",
                                default=False,
                                required=False,
                                help="Skip PR curve plotting"
                                )
    #############################################
    # Peaks subparser
    #############################################
    peaks_parser = subparsers.add_parser("peaks",
                                         parents=[parent_parser],
                                         help="Call peaks on a maxATAC prediction bigwig."
                                         )

    # Set the default function
    peaks_parser.set_defaults(func=run_call_peaks)

    # Add arguments to the parser
    peaks_parser.add_argument("-prefix", "--prefix",
                              dest="prefix",
                              type=str,
                              required=False,
                              help="Output prefix filename. Defaults: remove .bw extension."
                              )

    peaks_parser.add_argument("-bin", "--bin_size",
                              dest="BIN_SIZE",
                              type=int,
                              default=32,
                              help="Bin size to use for peak calling"
                              )

    peaks_parser.add_argument("-o", "--output",
                              dest="output",
                              type=str,
                              default="./peaks",
                              help="Output directory."
                              )

    peaks_parser.add_argument("-i", "--input_bigwig",
                              dest="input_bigwig",
                              type=str,
                              required=True,
                              help="Input bigwig"
                              )

    peaks_parser.add_argument("-cutoff_type", "--cutoff_type",
                              dest="cutoff_type",
                              default="F1",
                              type=str,
                              help="Cutoff type (i.e. Precision). Default: F1"
                              )

    peaks_parser.add_argument("-cutoff_value", "--cutoff_value",
                              dest="cutoff_value",
                              type=float,
                              help="Cutoff value for the cutoff type provided"
                              )

    peaks_parser.add_argument("-cutoff_file", "--cutoff_file",
                              required=True,
                              dest="cutoff_file",
                              type=str,
                              help="Cutoff file provided in /data/models"
                              )

    peaks_parser.add_argument("-chromosomes",
                              dest="chromosomes",
                              type=str,
                              nargs="+",
                              default=AUTOSOMAL_CHRS,
                              help="Chromosomes list for analysis. \
                              Optionally with regions in a form of chrN:start-end. \
                              Default: main human chromosomes, whole length"
                              )

    peaks_parser.add_argument("--loglevel",
                              dest="loglevel",
                              type=str,
                              default="info",
                              choices=LOG_LEVELS.keys(),
                              help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                              )

    #############################################
    # Variants subparser
    #############################################
    variants_parser = subparsers.add_parser("variants",
                                            parents=[parent_parser],
                                            help="Predict sequence specific transcription factor binding."
                                            )

    # Set the default function
    variants_parser.set_defaults(func=run_variants)

    # Add arguments to the parser
    variants_parser.add_argument("-m", "--model",
                                 dest="model",
                                 type=str,
                                 required=True,
                                 help="maxATAC model"
                                 )

    variants_parser.add_argument("-i", "-bw", "-signal", "--signal",
                                 dest="input_bigwig",
                                 type=str,
                                 required=True,
                                 help="Input ATAC-seq signal"
                                 )

    variants_parser.add_argument("-o", "--output",
                                 dest="output_dir",
                                 type=str,
                                 default="./variants",
                                 help="Output directory."
                                 )

    variants_parser.add_argument("-n", "--name", "--prefix",
                                 dest="name",
                                 type=str,
                                 required=True,
                                 help="Output filename without extension. Example: Tcell_chr1_rs1234_CTCF"
                                 )

    variants_parser.add_argument("--genome",
                                 dest="genome",
                                 type=str,
                                 default="hg38",
                                 required=False,
                                 help="The reference genome build to use."
                                 )

    variants_parser.add_argument("-s", "--sequence",
                                 dest="sequence",
                                 type=str,
                                 help="Input 2bit DNA sequence"
                                 )

    variants_parser.add_argument("-c", "-chroms", "--chromosomes",
                                 dest="chromosomes",
                                 nargs="+",
                                 default=ALL_CHRS,
                                 help="Chromosomes to limit prediction to"
                                 )

    variants_parser.add_argument("-variants_bed", "--variants_bed",
                                 dest="variants_bed",
                                 required=True,
                                 help="The variant start position in BED format with the nucleotide at that position"
                                 )

    variants_parser.add_argument("-roi", "--roi",
                                 dest="roi",
                                 required=False,
                                 help="A bed file of LD blocks to predict in specifically"
                                 )

    variants_parser.add_argument("--loglevel",
                                 dest="loglevel",
                                 type=str,
                                 default="info",
                                 choices=LOG_LEVELS.keys(),
                                 help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                 )

    variants_parser.add_argument("--blacklist",
                                 dest="blacklist",
                                 type=str,
                                 help="The blacklisted regions to exclude in bed format."
                                 )

    variants_parser.add_argument("-cs", "--chrom_sizes", "-chromosome_sizes",
                                 dest="chrom_sizes",
                                 type=str,
                                 help="Chrom sizes file. Default: hg38 chrom sizes"
                                 )

    variants_parser.add_argument("--step_size",
                                 dest="step_size",
                                 type=int,
                                 default=256,
                                 help="Step size to use to stagger prediction windows. Default: 256 bp (i.e. 1,024/4)"
                                 )

    #############################################
    # Prepare subparser
    #############################################
    prepare_parser = subparsers.add_parser("prepare",
                                           parents=[parent_parser],
                                           help="Prepare ATAC-seq data for maxATAC."
                                           )

    # Set the default function
    prepare_parser.set_defaults(func=run_prepare)

    # Add arguments to the parser
    prepare_parser.add_argument("-i", "--input",
                                dest="input",
                                type=str,
                                required=True,
                                help="Input BAM or scATAC fragments file"
                                )

    prepare_parser.add_argument("-o", "--output",
                                dest="output_dir",
                                type=str,
                                required=True,
                                help="Output directory path"
                                )

    prepare_parser.add_argument("-prefix", "--prefix", "-name", "-n",
                                dest="name",
                                type=str,
                                required=True,
                                help="Filename prefix to use as the basename"
                                )

    prepare_parser.add_argument("-cs", "--chrom_sizes", "--chromosome_sizes",
                                dest="chrom_sizes",
                                type=str,
                                help="Chrom sizes file. Default: hg38 chrom sizes"
                                )

    prepare_parser.add_argument("-slop", "--slop",
                                dest="slop",
                                type=int,
                                default=20,
                                help="The slop size to use around the Tn5 cut sites."
                                )

    prepare_parser.add_argument("-rpm", "--rpm_factor",
                                dest="rpm_factor",
                                type=int,
                                default=20000000,
                                help="The RPM factor to use for scaling your read depth normalized signal."
                                )

    prepare_parser.add_argument("--blacklist",
                                dest="blacklist",
                                type=str,
                                help="The blacklisted regions to exclude in bed format."
                                )

    prepare_parser.add_argument("--blacklist_bw",
                                dest="blacklist_bw",
                                type=str,
                                help="The blacklisted regions to exclude in bigwig format."
                                )

    prepare_parser.add_argument("-c", "-chroms", "--chromosomes",
                                dest="chromosomes",
                                type=str,
                                nargs="+",
                                default=AUTOSOMAL_CHRS,
                                help="The chromosomes to include in the final output."
                                )

    prepare_parser.add_argument("-t", "-threads", "--threads",
                                dest="threads",
                                type=int,
                                default=get_cpu_count(),
                                help="The number of threads to use"
                                )

    prepare_parser.add_argument("-skip_dedup", "--skip_deduplication",
                                dest="skip_dedup",
                                default=False,
                                action="store_true",
                                help="Whether to perform deduplication"
                                )

    prepare_parser.add_argument("--loglevel",
                                dest="loglevel",
                                type=str,
                                default="info",
                                choices=LOG_LEVELS.keys(),
                                help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                )
    #############################################
    # Threshold subparser
    #############################################
    threshold_parser = subparsers.add_parser("threshold",
                                             parents=[parent_parser],
                                             help="Generate model threshold statistics."
                                             )

    # Set the default function
    threshold_parser.set_defaults(func=run_thresholding)

    # Add arguments to the parser
    threshold_parser.add_argument("--prefix",
                                  dest="prefix",
                                  type=str,
                                  required=True,
                                  help="Output prefix."
                                  )

    threshold_parser.add_argument("--chrom_sizes",
                                  dest="chrom_sizes",
                                  type=str,
                                  help="Input chromosome sizes file. Default is hg38."
                                  )

    threshold_parser.add_argument("--chromosomes",
                                  dest="chromosomes",
                                  type=str,
                                  nargs="+",
                                  default=DEFAULT_VALIDATE_CHRS,
                                  help="Chromosomes for thresholding predictions. \
                                      Default: 1-22,X,Y"
                                  )

    threshold_parser.add_argument("--bin_size",
                                  dest="bin_size",
                                  type=int,
                                  default=DEFAULT_BENCHMARKING_BIN_SIZE,
                                  help="Chromosomes for averaging"
                                  )

    threshold_parser.add_argument("--output",
                                  dest="output_dir",
                                  type=str,
                                  default="./threshold",
                                  help="Output directory."
                                  )

    threshold_parser.add_argument("--loglevel",
                                  dest="loglevel",
                                  type=str,
                                  default="info",
                                  choices=LOG_LEVELS.keys(),
                                  help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                  )

    threshold_parser.add_argument("--blacklist_bw",
                                  dest="blacklist_bw",
                                  type=str,
                                  help="The blacklisted regions to exclude in bigwig format."
                                  )

    threshold_parser.add_argument("--meta_file",
                                  dest="meta_file",
                                  type=str,
                                  required=True,
                                  help="Meta file containing Prediction signal and GS path for all cell lines (.tsv format)"
                                  )

    return general_parser


def print_args(args, logger, header="Arguments:\n", excl=["func"]):
    """Print the arguments list
    """
    filtered = {
        k: v for k, v in args.__dict__.items()
        if k not in excl
    }
    logger(header + dump(filtered))


# we need to cwd_abs_path parameter only for running unit tests
def parse_arguments(argsl, cwd_abs_path=None):
    """Parse user arguments

    This function will parse, append, and modify the user inputs with consideration for reference genome. This function
    will also prepare the path names for easier testing.

    Args:
        argsl ([type]): list of user inputs
        cwd_abs_path ([type], optional): [description]. Defaults to None.

    Returns:
        Arguments list
    """
    cwd_abs_path = getcwd() if cwd_abs_path is None else cwd_abs_path

    if len(argsl) == 0:
        argsl.append("")  # otherwise fails with error if empty

    # Create the parser
    args, _ = get_parser().parse_known_args(argsl)

    # Update the reference genome paths
    args = update_reference_genome_paths(args)

    if args.func == run_training:
        args = normalize_args(
            args,
            [
                "func", "loglevel", "threads", "seed",
                "proportion", "vchroms", "tchroms", "multiprocessing",
                "chroms", "keep", "epochs", "batches", "max_queue_size",
                "prefix", "plot", "lrate", "decay", "bin",
                "minimum", "test_cell_lines", "rand_ratio",
                "train_tf", "arch", "batch_size", "save_roi",
                "val_batch_size", "target_scale_factor", "blacklist", "chrom_sizes",
                "output_activation", "dense", "shuffle_cell_type", "rev_comp", "genome"
            ],
            cwd_abs_path
        )

    return args
