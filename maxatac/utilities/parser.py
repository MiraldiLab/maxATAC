import argparse
import random
from yaml import dump

from maxatac.utilities.system_tools import (
    get_version,
    get_cpu_count
)

from maxatac.functions.train import run_training
from maxatac.functions.benchmark import run_benchmarking
from maxatac.functions.normalize import run_normalization
from maxatac.functions.predict import run_prediction
from maxatac.functions.average import run_averaging

from maxatac.utilities.constants import (
    LOG_LEVELS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_RAND_RATIO,
    DEFAULT_VALIDATE_RAND_RATIO,
    DEFAULT_TRAIN_BATCHES_PER_EPOCH,
    DEFAULT_VALIDATE_BATCHES_PER_EPOCH,
    DEFAULT_TRAIN_BATCH_SIZE,
    CHR_POOL_SIZE,
    BLACKLISTED_REGIONS,
    DEFAULT_BENCHMARKING_BIN_SIZE,
    INPUT_KERNEL_SIZE,
    INPUT_FILTERS,
    DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
    DEFAULT_ROUND,
    DEFAULT_MIN_PREDICTION,
    DEFAULT_TEST_CHRS,
    FILTERS_SCALING_FACTOR,
    DEFAULT_TRAIN_CHRS,
    DEFAULT_VALIDATE_CHRS,
    DEFAULT_CHROM_SIZES,
    DEFAULT_VALIDATE_BATCH_SIZE,
    DEFAULT_CHRS,
    BLACKLISTED_REGIONS_BIGWIG
)


def get_parser():
    # Parent parser for maxATAC
    parent_parser = argparse.ArgumentParser(add_help=False)

    # General parser for description and functions
    general_parser = argparse.ArgumentParser(description="maxATAC: \
        A suite of user-friendly, deep neural network models for \
        transcription factor binding prediction from ATAC-seq")

    # Add subparsers: to be used with different functions
    subparsers = general_parser.add_subparsers()

    # require subparsers
    subparsers.required = True

    # Add the version argument to the general parser.
    general_parser.add_argument(
        "--version", action="version", version=get_version(),
        help="Print version information and exit")

    # Average parser
    average_parser = subparsers.add_parser(
        "average",
        parents=[parent_parser],
        help="Run maxATAC average")

    # Set the default function to run averaging
    average_parser.set_defaults(func=run_averaging)

    average_parser.add_argument(
        "--bigwigs",
        dest="bigwig_files",
        type=str,
        nargs="+",
        required=True,
        help="Input bigwig files to average.")

    average_parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        required=True,
        help="Output prefix.")

    average_parser.add_argument(
        "--chrom_sizes",
        dest="chrom_sizes",
        type=str,
        default=DEFAULT_CHROM_SIZES,
        help="Input chromosome sizes file. Default is hg38.")

    average_parser.add_argument(
        "--chromosomes",
        dest="chromosomes",
        type=str,
        nargs="+",
        default=DEFAULT_CHRS,
        help="Chromosomes for averaging. \
                Default: 1-22,X,Y")

    average_parser.add_argument(
        "--output",
        dest="output_dir",
        type=str,
        default="./average",
        help="Output directory.")

    average_parser.add_argument(
        "--loglevel",
        dest="loglevel",
        type=str,
        default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL)

    # Predict parser
    predict_parser = subparsers.add_parser(
        "predict",
        parents=[parent_parser],
        help="Run maxATAC prediction")

    # Set the default function to run prediction
    predict_parser.set_defaults(func=run_prediction)

    predict_parser.add_argument(
        "--models",
        dest="models",
        type=str,
        nargs="+",
        required=True,
        help="Trained model file(s)")

    predict_parser.add_argument(
        "--sequence",
        dest="sequence",
        type=str,
        required=True,
        help="Genome sequence 2bit file")

    predict_parser.add_argument(
        "--signal",
        dest="signal",
        type=str,
        required=True,
        help="Input signal file")

    predict_parser.add_argument(
        "--round",
        dest="round",
        type=int,
        default=DEFAULT_ROUND,
        help="Float precision that you want to round predictions to")

    predict_parser.add_argument(
        "--minimum",
        dest="minimum",
        type=int,
        default=DEFAULT_MIN_PREDICTION,
        help="Minimum score threshold for predictions")

    predict_parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=1000,
        help="Number of regions to predict on at a time")

    predict_parser.add_argument(
        "--output_directory",
        dest="output_directory",
        type=str,
        default="./prediction_results",
        help="Folder for prediction results. Default: ./prediction_results")

    predict_parser.add_argument(
        "--threads",
        dest="threads",
        default=get_cpu_count(),
        type=int,
        help="# of processes to run prediction in parallel. \
            Default: # of --models multiplied by # of --chromosomes")

    predict_parser.add_argument(
        "--loglevel",
        dest="loglevel",
        type=str,
        default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL)

    predict_parser.add_argument(
        "--roi",
        dest="roi",
        type=str,
        required=True,
        help="Bed file with ranges for input sequences to be predicted. \
            Default: None, predictions are done on the whole chromosome length")

    predict_parser.add_argument(
        "--blacklist",
        dest="blacklist",
        type=str,
        default=BLACKLISTED_REGIONS,
        help="The blacklisted regions to exclude")

    predict_parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        default="maxatac_predict",
        help="Prefix for filename")

    predict_parser.add_argument(
        "--chromosome_sizes",
        dest="chromosome_sizes",
        type=str,
        default=DEFAULT_CHROM_SIZES,
        help="The chromosome sizes file to reference")

    predict_parser.add_argument(
        "--predict_chromosomes",
        dest="predict_chromosomes",
        type=str,
        nargs="+",
        default=DEFAULT_TEST_CHRS,
        help="Chromosomes from --chromosomes fixed for prediction. \
            Default: 1, 8")

    # Train parser
    train_parser = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        help="Run maxATAC training")

    # Set the default function to run training
    train_parser.set_defaults(func=run_training)

    train_parser.add_argument(
        "--sequence",
        dest="sequence",
        type=str,
        required=True,
        help="Genome sequence 2bit file")

    train_parser.add_argument(
        "--meta_file",
        dest="meta_file",
        type=str,
        required=True,
        help="Meta file containing ATAC Signal and Bindings path for all cell lines (.tsv format)")

    train_parser.add_argument(
        "--train_chroms",
        dest="train_chroms",
        type=str,
        nargs="+",
        default=DEFAULT_TRAIN_CHRS,
        help="Chromosomes from --chromosomes fixed for training. \
            Default: 3-7,9-18,20-22")

    train_parser.add_argument(
        "--validate_chroms",
        dest="validate_chroms",
        type=str,
        nargs="+",
        default=DEFAULT_VALIDATE_CHRS,
        help="Chromosomes from fixed for validation. \
            Default: chr2, chr19")

    train_parser.add_argument(
        "--arch",
        dest="arch",
        type=str,
        default="DCNN_V2",
        required=False,
        help="Specify the model architecture. Currently support DCNN_V2 or RES_DCNN_V2")

    train_parser.add_argument(
        "--train_rand_ratio",
        dest="train_rand_ratio",
        type=float,
        required=False,
        default=DEFAULT_TRAIN_RAND_RATIO,
        help="Ratio for controlling fraction of random sequences in each training batch. float [0, 1]")

    train_parser.add_argument(
        "--validate_rand_ratio",
        dest="validate_rand_ratio",
        type=float,
        required=False,
        default=DEFAULT_VALIDATE_RAND_RATIO,
        help="Ratio for controlling fraction of random seqeuences in each validation batch. float [0, 1]")

    train_parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=random.randint(1, 99999),
        help="Seed for pseudo-random generator. Default: random int [1, 99999]")

    train_parser.add_argument(
        "--weights",
        dest="weights",
        type=str,
        help="Weights to initialize model before training. Default: do not load")

    train_parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        default=DEFAULT_TRAIN_EPOCHS,
        help="# of training epochs. Default: " + str(DEFAULT_TRAIN_EPOCHS))

    train_parser.add_argument(
        "--FSF",
        dest="filter_scaling_factor",
        type=float,
        default=FILTERS_SCALING_FACTOR,
        help="Filter scaling factor. For each convolutional layer, multiply the number of filters by this argument. "
             "Default: " + str(FILTERS_SCALING_FACTOR))

    train_parser.add_argument(
        "--validate_batch_size",
        dest="validate_batch_size",
        type=int,
        default=DEFAULT_VALIDATE_BATCH_SIZE,
        help="# of validation examples per batch."
             "Default: " + str(DEFAULT_VALIDATE_BATCH_SIZE))

    train_parser.add_argument(
        "--validate_steps_per_epoch",
        dest="validate_steps_per_epoch",
        type=int,
        default=DEFAULT_VALIDATE_BATCHES_PER_EPOCH,
        help="# of validate batches per epoch."
             "Default: " + str(DEFAULT_TRAIN_BATCHES_PER_EPOCH))

    train_parser.add_argument(
        "--train_batch_size",
        dest="train_batch_size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="# of training examples per batch. \
            Default: " + str(DEFAULT_TRAIN_BATCH_SIZE))

    train_parser.add_argument(
        "--train_steps_per_epoch",
        dest="train_steps_per_epoch",
        type=int,
        default=DEFAULT_TRAIN_BATCHES_PER_EPOCH,
        help="# of training batches per epoch. \
            Default: " + str(DEFAULT_TRAIN_BATCHES_PER_EPOCH))

    train_parser.add_argument(
        "--filter_number",
        dest="number_of_filters",
        type=int,
        default=INPUT_FILTERS,
        help="# of filters to use for training. \
            Default: " + str(INPUT_FILTERS))

    train_parser.add_argument(
        "--kernel_size",
        dest="kernel_size",
        type=int,
        default=INPUT_KERNEL_SIZE,
        help="Size of the kernel to use in BP. \
            Default: " + str(INPUT_KERNEL_SIZE))

    train_parser.add_argument(
        "--chromosome_pool_size",
        dest="chromosome_pool_size",
        type=int,
        default=CHR_POOL_SIZE,
        help="Size of the kernel to use in BP. \
            Default: " + str(INPUT_KERNEL_SIZE))

    train_parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        default="maxATAC_model",
        help="Output prefix. Default: maxATAC_model")

    train_parser.add_argument(
        "--output",
        dest="output",
        type=str,
        default="./training_results",
        help="Folder for training results. Default: ./training_results")

    train_parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
        help="Plot model structure and training history. \
            Default: False")

    train_parser.add_argument(
        "--threads",
        dest="threads",
        default=get_cpu_count(),
        type=int,
        help="# of processes to run training in parallel. \
            Default: get the cpu count")

    train_parser.add_argument(
        "--loglevel",
        dest="loglevel",
        type=str,
        default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL)

    train_parser.add_argument(
        "--blacklist",
        dest="blacklist",
        type=str,
        default=BLACKLISTED_REGIONS,
        help="The blacklisted regions to exclude")

    train_parser.add_argument(
        "--chrom_sizes",
        dest="chrom_sizes",
        type=str,
        default=DEFAULT_CHROM_SIZES,
        help="The chromosome sizes file to reference")

    # Normalize parser
    normalize_parser = subparsers.add_parser(
        "normalize",
        parents=[parent_parser],
        help="Run minmax normalization")

    normalize_parser.set_defaults(func=run_normalization)

    normalize_parser.add_argument(
        "--signal",
        dest="signal",
        type=str,
        required=True,
        help="Input signal bigWig file(s) to be normalized by reference")

    normalize_parser.add_argument(
        "--chrom_sizes",
        dest="chrom_sizes",
        type=str,
        default=DEFAULT_CHROM_SIZES,
        help="Chrom sizes file")

    normalize_parser.add_argument(
        "--output",
        dest="output",
        type=str,
        default="./normalization_results",
        help="Folder for normalization results. Default: ./normalization_results")

    normalize_parser.add_argument(
        "--loglevel",
        dest="loglevel",
        type=str,
        default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL)

    # Benchmark parser
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        parents=[parent_parser],
        help="Run maxATAC benchmarking")

    # Set the default function to run benchmarking
    benchmark_parser.set_defaults(func=run_benchmarking)

    benchmark_parser.add_argument(
        "--prediction",
        dest="prediction",
        type=str,
        required=True,
        help="Prediction bigWig file")

    benchmark_parser.add_argument(
        "--gold_standard",
        dest="gold_standard",
        type=str,
        required=True,
        help="Gold Standard bigWig file")

    benchmark_parser.add_argument(
        "--chromosomes",
        dest="chromosomes",
        type=str,
        default=DEFAULT_TEST_CHRS,
        help="Chromosomes list for analysis. \
            Optionally with regions in a form of chrN:start-end. \
            Default: main human chromosomes, whole length")

    benchmark_parser.add_argument(
        "--bin_size",
        dest="bin_size",
        type=int,
        default=DEFAULT_BENCHMARKING_BIN_SIZE,
        help="Bin size to split prediction and control data before running prediction. \
            Default: " + str(DEFAULT_BENCHMARKING_BIN_SIZE))

    benchmark_parser.add_argument(
        "--agg",
        dest="agg_function",
        type=str,
        default=DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
        help="Aggregation function to use for combining results into bins: \
            max, sum, mean, median, min")

    benchmark_parser.add_argument(
        "--round_predictions",
        dest="round_predictions",
        type=int,
        default=DEFAULT_ROUND,
        help="Round binned values to this number of decimal places")

    benchmark_parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        required=True,
        help="Prefix for the file name")

    benchmark_parser.add_argument(
        "--output_directory",
        dest="output_directory",
        type=str,
        default="./benchmarking_results",
        help="Folder for benchmarking results. Default: ./benchmarking_results")

    benchmark_parser.add_argument(
        "--loglevel",
        dest="loglevel",
        type=str,
        default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL)

    benchmark_parser.add_argument(
        "--blacklist",
        dest="blacklist",
        type=str,
        default=BLACKLISTED_REGIONS_BIGWIG,
        help="The blacklisted regions to exclude")

    return general_parser


def print_args(args, logger, header="Arguments:\n", excl=["func"]):
    filtered = {
        k: v for k, v in args.__dict__.items()
        if k not in excl
    }

    logger(header + dump(filtered))


def parse_arguments(argsl):
    if len(argsl) == 0:
        argsl.append("")  # otherwise fails with error if empty

    args, _ = get_parser().parse_known_args(argsl)

    return args
