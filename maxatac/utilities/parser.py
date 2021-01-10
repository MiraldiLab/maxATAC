import argparse
import random
from os import getcwd
from yaml import dump

from maxatac.utilities.system_tools import (
    get_version,
    get_cpu_count
)

from maxatac.functions.train import run_training
from maxatac.functions.benchmark import run_benchmarking
from maxatac.functions.normalize import run_normalization
from maxatac.functions.predict import run_prediction

from maxatac.utilities.constants import (
    LOG_LEVELS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_BATCHES_PER_EPOCH,
    BATCH_SIZE,
    CHR_POOL_SIZE,
    BLACKLISTED_REGIONS,
    DEFAULT_BENCHMARKING_BIN_SIZE,
    INPUT_KERNEL_SIZE,
    INPUT_FILTERS,
    DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
    DEFAULT_ROUND,
    DEFAULT_PREDICTION_BATCH_SIZE,
    DEFAULT_TEST_CHRS,
    FILTERS_SCALING_FACTOR,
    DEFAULT_TRAIN_CHRS,
    DEFAULT_VALIDATE_CHRS,
    DEFAULT_CHROM_SIZES
)


def get_parser():
    # Parent (general) parser
    parent_parser = argparse.ArgumentParser(add_help=False)

    general_parser = argparse.ArgumentParser(description="maxATAC: \
        DeepCNN for predicting TF binding from ATAC-seq")

    subparsers = general_parser.add_subparsers()

    subparsers.required = True

    general_parser.add_argument(
        "--version", action="version", version=get_version(),
        help="Print version information and exit")

    # Predict parser
    predict_parser = subparsers.add_parser(
        "predict",
        parents=[parent_parser],
        help="Run maxATAC prediction")

    predict_parser.set_defaults(func=run_prediction)

    predict_parser.add_argument(
        "--models",
        dest="models",
        type=str,
        nargs="+",
        required=True,
        help="Trained model file(s)")

    predict_parser.add_argument(
        "--average",
        dest="average",
        type=str,
        required=True,
        help="Average signal bigWig file")

    predict_parser.add_argument(
        "--sequence",
        dest="sequence",
        type=str,
        required=True,
        help="Genome sequence 2bit file")

    predict_parser.add_argument(
        "--round",
        dest="round",
        type=int,
        default=DEFAULT_ROUND,
        help="Float precision that you want to round predictions to")

    predict_parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=DEFAULT_PREDICTION_BATCH_SIZE,
        help="Float precision that you want to round predictions to")

    predict_parser.add_argument(
        "--tmp",
        dest="tmp",
        type=str,
        default="./temp",
        help="Folder to save temporary data. Default: ./temp")

    predict_parser.add_argument(
        "--output",
        dest="output",
        type=str,
        default="./prediction_results",
        help="Folder for prediction results. Default: ./prediction_results")

    predict_parser.add_argument(
        "--keep",
        dest="keep",
        action="store_true",
        help="Keep temporary files. Default: False")

    predict_parser.add_argument(
        "--threads",
        dest="threads",
        default=get_cpu_count(),
        type=int,
        help="# of processes to run prediction in parallel. \
            Default: # of --models multiplied by # of --chroms")

    predict_parser.add_argument(
        "--loglevel",
        dest="loglevel",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL)

    predict_parser.add_argument(
        "--predict_roi",
        dest="predict_roi",
        type=str,
        help="Bed file with ranges for input sequences to be predicted. \
            Default: None, predictions are done on the whole chromosome length")

    # Train parser
    train_parser = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        help="Run maxATAC training")

    train_parser.set_defaults(func=run_training)

    train_parser.add_argument(
        "--sequence",
        dest="sequence",
        type=str,
        required=True,
        help="Genome sequence 2bit file")

    train_parser.add_argument(
        "--average",
        dest="average",
        type=str,
        required=True,
        help="Average signal bigWig file")

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
        help="Chromosomes from --chroms fixed for training. \
            Default: 3-7,9-18,20-22")

    train_parser.add_argument(
        "--validate_chroms",
        dest="validate_chroms",
        type=str,
        nargs="+",
        default=DEFAULT_VALIDATE_CHRS,
        help="Chromosomes from --chroms fixed for validation. \
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
        default=.5,
        help="Ratio for controlling fraction of random seqeuences in each training batch. float [0, 1]")

    train_parser.add_argument(
        "--validate_rand_ratio",
        dest="validate_rand_ratio",
        type=float,
        required=False,
        default=.5,
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
        dest="FilterScalingFactor",
        type=float,
        default=FILTERS_SCALING_FACTOR,
        help="Filter scaling factor. For each convolutional layer, multiply the number of filters by this argument. "
             "Default: " + str(FILTERS_SCALING_FACTOR))

    train_parser.add_argument(
        "--validate_batch_size",
        dest="validate_batch_size",
        type=int,
        default=BATCH_SIZE,
        help="# of validation examples per batch."
             "Default: " + str(BATCH_SIZE))

    train_parser.add_argument(
        "--validate_steps_per_epoch",
        dest="validate_steps_per_epoch",
        type=int,
        default=DEFAULT_TRAIN_BATCHES_PER_EPOCH,
        help="# of validate batches per epoch."
             "Default: " + str(DEFAULT_TRAIN_BATCHES_PER_EPOCH))

    train_parser.add_argument(
        "--train_batch_size",
        dest="train_batch_size",
        type=int,
        default=BATCH_SIZE,
        help="# of training examples per batch. \
            Default: " + str(BATCH_SIZE))

    train_parser.add_argument(
        "--train_steps_per_epoch",
        dest="train_steps_per_epoch",
        type=int,
        default=DEFAULT_TRAIN_BATCHES_PER_EPOCH,
        help="# of training batches per epoch. \
            Default: " + str(DEFAULT_TRAIN_BATCHES_PER_EPOCH))

    train_parser.add_argument(
        "--filter_number",
        dest="FilterNumber",
        type=int,
        default=INPUT_FILTERS,
        help="# of filters to use for training. \
            Default: " + str(INPUT_FILTERS))

    train_parser.add_argument(
        "--kernel_size",
        dest="KernelSize",
        type=int,
        default=INPUT_KERNEL_SIZE,
        help="Size of the kernel to use in BP. \
            Default: " + str(INPUT_KERNEL_SIZE))

    train_parser.add_argument(
        "--chrom_pool_size",
        dest="chrom_pool_size",
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
        default=1,
        type=int,
        help="# of processes to run training in parallel. \
            Default: 1")

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
        help="The chrom sizes file to reference")

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
        "--genome",
        dest="GENOME",
        type=str,
        default=DEFAULT_CHROM_SIZES,
        help="Reference genome build")

    normalize_parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        required=True,
        default="normalized",
        help="Output prefix. Default: normalized")

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
        default=DEFAULT_LOG_LEVEL,
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL)

    # Benchmark parser
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        parents=[parent_parser],
        help="Run maxATAC benchmarking")

    benchmark_parser.set_defaults(func=run_benchmarking)

    benchmark_parser.add_argument(
        "--prediction",
        dest="prediction",
        type=str,
        required=True,
        help="Prediction bigWig file")

    benchmark_parser.add_argument(
        "--goldstandard",
        dest="goldstandard",
        type=str,
        required=True,
        help="Gold Standard bigWig file")

    benchmark_parser.add_argument(
        "--chroms",
        dest="chroms",
        type=str,
        default=DEFAULT_TEST_CHRS,
        help="Chromosomes list for analysis. \
            Optionally with regions in a form of chrN:start-end. \
            Default: main human chromosomes, whole length")

    benchmark_parser.add_argument(
        "--bin",
        dest="bin",
        type=int,
        default=DEFAULT_BENCHMARKING_BIN_SIZE,
        help="Bin size to split prediction and control data before running prediction. \
            Default: " + str(DEFAULT_BENCHMARKING_BIN_SIZE))

    benchmark_parser.add_argument(
        "--agg",
        dest="agg_function",
        type=int,
        default=DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
        help="Aggregation function to use for combining results into bins: \
            max, sum, mean, median, min")

    benchmark_parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        required=True,
        help="Prefix for the file name")

    benchmark_parser.add_argument(
        "--output",
        dest="output",
        type=str,
        default="./benchmarking_results",
        help="Folder for benchmarking results. Default: ./benchmarking_results")

    benchmark_parser.add_argument(
        "--loglevel",
        dest="loglevel",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL)

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

    return args
