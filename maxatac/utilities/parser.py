import argparse
import random
from os import getcwd
from pkg_resources import require

from yaml import dump

from maxatac.utilities.system_tools import (get_version,
                                            get_absolute_path,
                                            get_cpu_count,
                                            Mute
                                            )

with Mute():
    from maxatac.analyses.average import run_averaging
    from maxatac.analyses.predict import run_prediction
    from maxatac.analyses.train import run_training
    from maxatac.analyses.normalize import run_normalization
    from maxatac.analyses.benchmark import run_benchmarking
    from maxatac.analyses.prediction_signal import run_prediction_signal
    from maxatac.analyses.peaks import run_call_peaks
    from maxatac.analyses.variants import run_variants
    from maxatac.analyses.prepare import run_prepare
    
from maxatac.utilities.constants import (DEFAULT_TRAIN_VALIDATE_CHRS,
                                         LOG_LEVELS,
                                         DEFAULT_LOG_LEVEL,
                                         DEFAULT_TRAIN_EPOCHS,
                                         DEFAULT_TRAIN_BATCHES_PER_EPOCH,
                                         DEFAULT_MIN_PREDICTION,
                                         BATCH_SIZE,
                                         VAL_BATCH_SIZE,
                                         INPUT_LENGTH,
                                         DEFAULT_TRAIN_CHRS,
                                         DEFAULT_VALIDATE_CHRS,
                                         DEFAULT_CHROM_SIZES,
                                         BLACKLISTED_REGIONS,
                                         DEFAULT_VALIDATE_RAND_RATIO,
                                         DEFAULT_ROUND,
                                         DEFAULT_TEST_CHRS,
                                         BLACKLISTED_REGIONS_BIGWIG,
                                         DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
                                         DEFAULT_BENCHMARKING_BIN_SIZE,
                                         ALL_CHRS, AUTOSOMAL_CHRS
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
    # Parent (general) parser
    parent_parser = argparse.ArgumentParser(add_help=False)
    general_parser = argparse.ArgumentParser(description="Neural networks for predicting TF binding using ATAC-seq")
    subparsers = general_parser.add_subparsers()
    subparsers.required = True

    general_parser.add_argument("--version",
                                action="version",
                                version=get_version(),
                                help="Print version information and exit"
                                )

    #############################################

    # Average parser
    average_parser = subparsers.add_parser("average",
                                           parents=[parent_parser],
                                           help="Run maxATAC average"
                                           )

    # Set the default function to run averaging
    average_parser.set_defaults(func=run_averaging)

    average_parser.add_argument("--bigwigs",
                                dest="bigwig_files",
                                type=str,
                                nargs="+",
                                required=True,
                                help="Input bigwig files to average."
                                )

    average_parser.add_argument("--prefix",
                                dest="prefix",
                                type=str,
                                required=True,
                                help="Output filename prefix."
                                )

    average_parser.add_argument("--chrom_sizes",
                                dest="chrom_sizes",
                                type=str,
                                default=DEFAULT_CHROM_SIZES,
                                help="Input chromosome sizes file. Default is hg38."
                                )

    average_parser.add_argument("--chromosomes",
                                dest="chromosomes",
                                type=str,
                                nargs="+",
                                default=AUTOSOMAL_CHRS,
                                help="Chromosomes for averaging. \
                                      Default: 1-22"
                                )

    average_parser.add_argument("--output",
                                dest="output_dir",
                                type=str,
                                default="./average",
                                help="Output directory."
                                )

    average_parser.add_argument("--loglevel",
                                dest="loglevel",
                                type=str,
                                default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                                choices=LOG_LEVELS.keys(),
                                help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                )

    #############################################

    # Predict parser
    predict_parser = subparsers.add_parser("predict",
                                           parents=[parent_parser],
                                           help="Run maxATAC prediction",
                                           )

    predict_parser.set_defaults(func=run_prediction)

    predict_parser.add_argument("--models", dest="models", type=str, nargs="+",
                                required=True,
                                help="Trained model file(s)"
                                )

    predict_parser.add_argument("--sequence",
                                dest="sequence",
                                type=str,
                                required=True,
                                help="Genome sequence 2bit file"
                                )

    predict_parser.add_argument("--signal",
                                dest="signal",
                                type=str,
                                required=True,
                                help="Input signal file"
                                )

    predict_parser.add_argument("--output",
                                dest="output",
                                type=str,
                                default="./prediction_results",
                                help="Folder for prediction results. Default: ./prediction_results"
                                )

    predict_parser.add_argument("--blacklist",
                                dest="blacklist",
                                type=str,
                                default=BLACKLISTED_REGIONS,
                                help="The blacklisted regions to exclude"
                                )

    predict_parser.add_argument("--roi",
                                dest="roi",
                                default=False,
                                required=False,
                                help="Bed file with ranges for input sequences to be predicted. \
                                      Default: None, predictions are done on the whole chromosome length"
                                )

    predict_parser.add_argument("--stranded",
                                dest="stranded",
                                default=False,
                                action='store_true',
                                required=False,
                                help="Whether to make predictions based on both strands")

    predict_parser.add_argument("--threads",
                                dest="threads",
                                default=get_cpu_count(),
                                type=int,
                                help="# of processes to run prediction in parallel. \
                                        Default: # of --models multiplied by # of --chromosomes"
                                )

    predict_parser.add_argument("--loglevel",
                                dest="loglevel",
                                type=str,
                                default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
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
                                default=INPUT_LENGTH,
                                help="Step size to use to build sliding window regions"
                                )

    predict_parser.add_argument("--prefix",
                                dest="prefix",
                                type=str,
                                default="maxatac_predict",
                                help="Prefix for filename"
                                )

    predict_parser.add_argument("--chromosome_sizes",
                                dest="chromosome_sizes",
                                type=str,
                                default=DEFAULT_CHROM_SIZES,
                                help="The chromosome sizes file to reference"
                                )

    predict_parser.add_argument("--chromosomes",
                                dest="chromosomes",
                                type=str,
                                nargs="+",
                                default=DEFAULT_TEST_CHRS,
                                help="Chromosomes from --chromosomes fixed for prediction. \
                                      Default: 1, 8"
                                )
    predict_parser.add_argument("-bin", "--bin_size",
                              dest="BIN_SIZE",
                              type=int,
                              default=DEFAULT_BENCHMARKING_BIN_SIZE,
                              help="Bin size to use for peak calling")

    predict_parser.add_argument("-cutoff_type", "--cutoff_type",
                              dest="cutoff_type",
                              type=str,
                              help="Cutoff type (i.e. Precision)")

    predict_parser.add_argument("-cutoff_value", "--cutoff_value",
                              dest="cutoff_value",
                              type=float,
                              help="Cutoff value for the cutoff type provided")

    predict_parser.add_argument("-cutoff_file", "--cutoff_file",
                              dest="cutoff_file",
                              type=str,
                              help="Cutoff file provided in /data/models")

    #############################################

    # Train parser
    train_parser = subparsers.add_parser("train",
                                         parents=[parent_parser],
                                         help="Run maxATAC training"
                                         )

    train_parser.set_defaults(func=run_training)

    train_parser.add_argument("--sequence",
                              dest="sequence",
                              type=str,
                              required=True,
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

    # I set default to sigmoid.
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
                              default=.3,
                              help="Ratio for controlling fraction of random sequences in each training batch. "
                                   "Default: .3 "
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
                              default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                              choices=LOG_LEVELS.keys(),
                              help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                              )

    train_parser.add_argument("--shuffle_cell_type",
                              dest="shuffle_cell_type",
                              action="store_true",
                              default=False,
                              help="If shuffle_cell_type, then shuffle training ROI cell type label"
                              )

    train_parser.add_argument("--rev_comp",
                              dest="rev_comp",
                              action="store_true",
                              default=False,
                              help="If rev_comp, then use the reverse complement in training"
                              )

    #############################################

    # Normalize parser
    normalize_parser = subparsers.add_parser("normalize",
                                             parents=[parent_parser],
                                             help="Run minmax normalization")

    normalize_parser.set_defaults(func=run_normalization)

    normalize_parser.add_argument("--signal",
                                  dest="signal",
                                  type=str,
                                  required=True,
                                  help="Input signal bigWig file(s) to be normalized by reference")

    normalize_parser.add_argument("--chrom_sizes",
                                  dest="chrom_sizes",
                                  type=str,
                                  default=DEFAULT_CHROM_SIZES,
                                  help="Chrom sizes file")

    normalize_parser.add_argument("--chroms",
                                  dest="chroms",
                                  type=str,
                                  nargs="+",
                                  default=AUTOSOMAL_CHRS,
                                  help="Chromosome list for analysis. \
                                    Regions in a form of chrN:start-end are ignored. \
                                    Use --filters instead \
                                    Default: main human chromosomes, whole length"
                                  )

    normalize_parser.add_argument("--output",
                                  dest="output",
                                  type=str,
                                  default="./normalize",
                                  help="Folder for normalization results. Default: ./normalization_results")

    normalize_parser.add_argument("--prefix",
                                  dest="prefix",
                                  type=str,
                                  default="normalized",
                                  help="Name to use for filename")

    normalize_parser.add_argument("--min",
                                  dest="min",
                                  required=False,
                                  type=int,
                                  default=0,
                                  help="The minimum value to use for normalization")

    normalize_parser.add_argument("--max",
                                  dest="max",
                                  type=int,
                                  required=False,
                                  default=False,
                                  help="The maximum value to use for normalization")

    normalize_parser.add_argument("--clip",
                                  dest="clip",
                                  type=bool,
                                  required=False,
                                  default=False,
                                  help="Whether to clip minmax values to the range 0,1")

    normalize_parser.add_argument("--method",
                                  dest="method",
                                  type=str,
                                  default="min-max",
                                  help="The method to use for normalization")

    normalize_parser.add_argument("--max_percentile",
                                  dest="max_percentile",
                                  required=False,
                                  type=int,
                                  default=100,
                                  help="The maximum percentile to use for normalization")

    normalize_parser.add_argument("--loglevel",
                                  dest="loglevel",
                                  type=str,
                                  default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                                  choices=LOG_LEVELS.keys(),
                                  help="Logging level. Default: " + DEFAULT_LOG_LEVEL)

    normalize_parser.add_argument("--blacklist",
                                  dest="blacklist",
                                  type=str,
                                  default=BLACKLISTED_REGIONS_BIGWIG,
                                  help="The blacklisted regions to exclude"
                                  )

    #############################################

    # Benchmark parser
    benchmark_parser = subparsers.add_parser("benchmark",
                                             parents=[parent_parser],
                                             help="Run maxATAC benchmarking"
                                             )

    benchmark_parser.set_defaults(func=run_benchmarking)

    benchmark_parser.add_argument("--prediction",
                                  dest="prediction",
                                  type=str,
                                  required=True,
                                  help="Prediction bigWig file"
                                  )

    benchmark_parser.add_argument("--gold_standard",
                                  dest="gold_standard",
                                  type=str,
                                  required=True,
                                  help="Gold Standard bigWig file"
                                  )

    benchmark_parser.add_argument("--chromosomes",
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
                                        max, coverage, mean, std, min"
                                  )

    benchmark_parser.add_argument("--round_predictions",
                                  dest="round_predictions",
                                  type=int,
                                  default=DEFAULT_ROUND,
                                  help="Round binned values to this number of decimal places"
                                  )

    benchmark_parser.add_argument("--prefix",
                                  dest="prefix",
                                  type=str,
                                  required=True,
                                  help="Prefix for the file name"
                                  )

    benchmark_parser.add_argument("--output_directory",
                                  dest="output_directory",
                                  type=str,
                                  default="./benchmarking_results",
                                  help="Folder for benchmarking results. Default: ./benchmarking_results"
                                  )

    benchmark_parser.add_argument("--loglevel",
                                  dest="loglevel",
                                  type=str,
                                  default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                                  choices=LOG_LEVELS.keys(),
                                  help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                  )

    benchmark_parser.add_argument("--blacklist",
                                  dest="blacklist",
                                  type=str,
                                  default=BLACKLISTED_REGIONS_BIGWIG,
                                  help="The blacklisted regions to exclude"
                                  )

    #############################################

    # Prediction_signal parser
    prediction_signal_parser = subparsers.add_parser("prediction_signal",
                                                     parents=[parent_parser],
                                                     help="Run maxATAC prediction_signal"
                                                     )

    prediction_signal_parser.set_defaults(func=run_prediction_signal)

    prediction_signal_parser.add_argument("--prediction",
                                          dest="prediction",
                                          type=str,
                                          required=True,
                                          help="Prediction bigWig file"
                                          )

    prediction_signal_parser.add_argument("--sequence",
                                          dest="sequence",
                                          type=str,
                                          required=True,
                                          help="hg38 sequence file"
                                          )

    prediction_signal_parser.add_argument("--chromosomes",
                                          dest="chromosomes",
                                          type=str,
                                          nargs="+",
                                          default=DEFAULT_TEST_CHRS,
                                          help="Chromosomes list for analysis. \
                                            Optionally with regions in a form of chrN:start-end. \
                                            Default: main human chromosomes, whole length"
                                          )

    prediction_signal_parser.add_argument("--bin_size",
                                          dest="bin_size",
                                          type=int,
                                          default=DEFAULT_BENCHMARKING_BIN_SIZE,
                                          help="Bin size to split prediction and control data before running prediction. \
                                            Default: " + str(DEFAULT_BENCHMARKING_BIN_SIZE)
                                          )

    prediction_signal_parser.add_argument("--agg",
                                          dest="agg_function",
                                          type=str,
                                          default=DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
                                          help="Aggregation function to use for combining results into bins: \
                                            max, coverage, mean, std, min"
                                          )

    prediction_signal_parser.add_argument("--round_predictions",
                                          dest="round_predictions",
                                          type=int,
                                          default=DEFAULT_ROUND,
                                          help="Round binned values to this number of decimal places"
                                          )

    prediction_signal_parser.add_argument("--prefix",
                                          dest="prefix",
                                          type=str,
                                          required=True,
                                          help="Prefix for the file name"
                                          )

    prediction_signal_parser.add_argument("--output_directory",
                                          dest="output_directory",
                                          type=str,
                                          default="./benchmarking_results",
                                          help="Folder for benchmarking results. Default: ./benchmarking_results"
                                          )

    prediction_signal_parser.add_argument("--loglevel",
                                          dest="loglevel",
                                          type=str,
                                          default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                                          choices=LOG_LEVELS.keys(),
                                          help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                          )

    prediction_signal_parser.add_argument("--blacklist",
                                          dest="blacklist",
                                          type=str,
                                          default=BLACKLISTED_REGIONS_BIGWIG,
                                          help="The blacklisted regions to exclude"
                                          )

    #############################################

    # peaks_parser
    peaks_parser = subparsers.add_parser("peaks",
                                         parents=[parent_parser],
                                         help="Run maxATAC peaks"
                                         )

    # Set the default function to run averaging
    peaks_parser.set_defaults(func=run_call_peaks)

    peaks_parser.add_argument("-prefix", "--prefix",
                              dest="prefix",
                              type=str,
                              required=False,
                              help="Output prefix filename. Defaults: remove .bw extension."
                              )

    peaks_parser.add_argument("-bin", "--bin_size",
                              dest="BIN_SIZE",
                              type=int,
                              default=DEFAULT_BENCHMARKING_BIN_SIZE,
                              help="Bin size to use for peak calling")

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
                              help="Input bigwig")

    peaks_parser.add_argument("-cutoff_type", "--cutoff_type",
                              dest="cutoff_type",
                              type=str,
                              help="Cutoff type (i.e. Precision)")

    peaks_parser.add_argument("-cutoff_value", "--cutoff_value",
                              dest="cutoff_value",
                              type=float,
                              help="Cutoff value for the cutoff type provided")

    peaks_parser.add_argument("-cutoff_file", "--cutoff_file",
                              dest="cutoff_file",
                              type=str,
                              help="Cutoff file provided in /data/models")

    peaks_parser.add_argument("--chromosomes",
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
                              default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                              choices=LOG_LEVELS.keys(),
                              help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                              )

    #############################################

    # variants_parser
    variants_parser = subparsers.add_parser("variants",
                                         parents=[parent_parser],
                                         help="Run maxATAC variants"
                                         )

    # Set the default function to run variants
    variants_parser.set_defaults(func=run_variants)

    variants_parser.add_argument("-m", "--model",
                              dest="model",
                              type=str,
                              required=True,
                              help="maxATAC model"
                              )

    variants_parser.add_argument("-signal", "--signal",
                              dest="input_bigwig",
                              type=str,
                              required=True,
                              help="Input ATAC-seq signal")

    variants_parser.add_argument("-o", "--output",
                              dest="output",
                              type=str,
                              default="./variants",
                              help="Output directory."
                              )

    variants_parser.add_argument("-n", "--name",
                              dest="name",
                              type=str,
                              required=True,
                              help="Output filename without extension. Example: Tcell_chr1_rs1234_CTCF"
                              )
    
    variants_parser.add_argument("-s", "--sequence",
                              dest="sequence",
                              required=True,
                              type=str,
                              help="Input 2bit DNA sequence")

    variants_parser.add_argument("-chrom", "--chromosome",
                              dest="chromosome",
                              required=True,
                              help="Chromosome name")

    variants_parser.add_argument("-p", "--position",
                              dest="variant_start_pos",
                              type=int,
                              required=True,
                              help="The variant start position. This is the position where the variant is located in 0-based coordinates"
                              )

    variants_parser.add_argument("-nuc", "--target_nucleotide",
                              dest="nucleotide",
                              type=str,
                              required=True,
                              help="The nucldeotide to use at the variant start position. Example: A")
    
    variants_parser.add_argument("-overhang",
                              dest="overhang",
                              type=int,
                              default=0,
                              help="The amount of overhang around the 1,024 bp prediction window. Must be in intervals of 256 base pairs. Example: 512")

    variants_parser.add_argument("--loglevel",
                              dest="loglevel",
                              type=str,
                              default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                              choices=LOG_LEVELS.keys(),
                              help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                              )

    #############################################
    
    # prepare_parser
    prepare_parser = subparsers.add_parser("prepare",
                                         parents=[parent_parser],
                                         help="Run maxATAC prepare"
                                         )

    # Set the default function to run variants
    prepare_parser.set_defaults(func=run_prepare)

    prepare_parser.add_argument("-i", "--input",
                              dest="input",
                              type=str,
                              required=True,
                              help="Input BAM or scATAC fragments file"
                              )

    prepare_parser.add_argument("-o", "--output",
                              dest="output",
                              type=str,
                              required=True,
                              help="Output directory path"
                              )

    prepare_parser.add_argument("-prefix", "--prefix",
                              dest="prefix",
                              type=str,
                              required=True,
                              help="Filename prefix to use as the basename"
                              )
    
    prepare_parser.add_argument("--chrom_sizes",
                                  dest="chrom_sizes",
                                  type=str,
                                  default=DEFAULT_CHROM_SIZES,
                                  help="Chrom sizes file")

    prepare_parser.add_argument("-slop", "--slop",
                              dest="slop",
                              type=int,
                              default=20,
                              help="Slop size to use with cut sites"
                              )
    
    prepare_parser.add_argument("-rpm", "--rpm_factor",
                            dest="rpm_factor",
                            type=int,
                            default=20000000,
                            help="What millions value to scale data to"
                            )
    
    prepare_parser.add_argument("--blacklist_bed",
                                dest="blacklist_bed",
                                type=str,
                                default=BLACKLISTED_REGIONS,
                                help="The blacklisted regions to exclude"
                                )

    prepare_parser.add_argument("--blacklist",
                                dest="blacklist",
                                type=str,
                                default=BLACKLISTED_REGIONS_BIGWIG,
                                help="The blacklisted regions to exclude"
                                )
    
    prepare_parser.add_argument("-chroms", "--chromosomes",
                                        dest="chroms",
                                        type=str,
                                        nargs="+",
                                        default=AUTOSOMAL_CHRS,
                                        help="Chromosomes list for analysis."
                                        )

    prepare_parser.add_argument("-threads", "--threads",
                                        dest="threads",
                                        type=int,
                                        default=get_cpu_count(),
                                        help="The number of threads to use"
                                        )

    prepare_parser.add_argument("-dedup", "--deduplicate",
                                        dest="dedup",
                                        default=False,
                                        action="store_true",
                                        help="Whether to perform deduplication"
                                )
        
    prepare_parser.add_argument("--loglevel",
                              dest="loglevel",
                              type=str,
                              default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                              choices=LOG_LEVELS.keys(),
                              help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                              )
    return general_parser


def print_args(args, logger, header="Arguments:\n", excl=["func"]):
    filtered = {
        k: v for k, v in args.__dict__.items()
        if k not in excl
    }
    logger(header + dump(filtered))


# we need to cwd_abs_path parameter only for running unit tests
def parse_arguments(argsl, cwd_abs_path=None):
    cwd_abs_path = getcwd() if cwd_abs_path is None else cwd_abs_path
    if len(argsl) == 0:
        argsl.append("")  # otherwise fails with error if empty
    args, _ = get_parser().parse_known_args(argsl)

    if args.func == run_training:
        args = normalize_args(
            args,
            [
                "func", "loglevel", "threads", "seed",
                "proportion", "vchroms", "tchroms",
                "chroms", "keep", "epochs", "batches",
                "prefix", "plot", "lrate", "decay", "bin",
                "minimum", "test_cell_lines", "rand_ratio",
                "train_tf", "arch", "batch_size",
                "val_batch_size", "target_scale_factor",
                "output_activation", "dense", "shuffle_cell_type", "rev_comp"
            ],
            cwd_abs_path
        )

    return args
