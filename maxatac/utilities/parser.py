import argparse
import random
from os import path, getcwd
from time import strftime
from uuid import uuid4
from yaml import dump

from maxatac.analyses.prediction import run_prediction
from maxatac.analyses.training import run_training
from maxatac.analyses.normalize import run_normalization
from maxatac.analyses.benchmark import run_benchmarking

from maxatac.utilities.helpers import (
    get_version,
    get_absolute_path,
    get_cpu_count
)
from maxatac.utilities.constants import (
    DEFAULT_CHRS,
    DEFAULT_CHR_PROPORTION,
    LOG_LEVELS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_BATCHES_PER_EPOCH,
    DEFAULT_ADAM_LEARNING_RATE,
    DEFAULT_ADAM_DECAY,
    DEFAULT_NORMALIZATION_BIN,
    DEFAULT_MIN_PREDICTION,
    BATCH_SIZE,
    VAL_BATCH_SIZE
)
from maxatac.utilities.bigwig import load_bigwig
from maxatac.utilities.twobit import load_2bit




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


def get_synced_chroms(chroms, files, ignore_regions=None):
    """
    If ignore_regions is True, set regions to the whole chromosome length
    Returns something like this
        {
            "chr2": {"length": 243199373, "region": [0, 243199373]},
            "chr3": {"length": 198022430, "region": [0, 198022430]}
        }
    """
    
    chroms_and_regions = {}
    for chrom in chroms:
        chrom_name, *region = chrom.replace(",", "").split(":")  # region is either [] or ["start-end", ...]
        chroms_and_regions[chrom_name] = None
        if not ignore_regions:
            try:
                chroms_and_regions[chrom_name] = [int(i) for i in region[0].split("-")]
            except (IndexError, ValueError):
                pass

    loaded_chroms = set()
    
    for file in [f for f in files if f is not None]:
        try:
            with load_2bit(file) as data_stream:
                avail_chroms = set([(k, v) for k, v in data_stream.chroms().items()])
        except RuntimeError:
            with load_bigwig(file) as data_stream:
                avail_chroms = set([(k, v) for k, v in data_stream.chroms().items()])
        loaded_chroms = loaded_chroms.intersection(avail_chroms) if loaded_chroms else avail_chroms  # checks both chrom_name and chrom_length are the same
    
    
    synced_chroms = {}
    for chrom_name, chrom_length in loaded_chroms:
        if chrom_name not in chroms_and_regions: continue
        region = chroms_and_regions[chrom_name]
        if not region or \
               region[0] < 0 or \
               region[1] <= 0 or \
               region[0] >= region[1] or \
               region[1] > chrom_length:
            region = [0, chrom_length]
        synced_chroms[chrom_name] = {
            "length": chrom_length,
            "region": region
        }
    
    return synced_chroms


def assert_and_fix_args_for_prediction(args):

    synced_chroms = get_synced_chroms(
        args.chroms,
        [args.sequence, args.signal, args.average]
    )
    assert len(synced_chroms) > 0, \
        "Failed to sync chromosomes. Check --chroms. Exiting"
    setattr(args, "chroms", synced_chroms)

    args.tmp = path.join(
        args.tmp,
        "-".join([strftime("%Y-%m-%d-%H-%M-%S"), str(uuid4())]))

    if args.threads is None:
        args.threads = len(args.models) * len(args.chroms)  # TODO: make sure that it's not more than available core count in GPU, or we are not using CPU


def assert_and_fix_args_for_training(args):

    setattr(args, "preferences", None)
    setattr(args, "signal", None)
    #setattr(args, "filters", None)
    setattr(args, "tsites", None)
    synced_tchroms = get_synced_chroms(
        args.tchroms,
        [
            args.sequence,
            args.average,
            #args.filters,
            #args.preferences,
            #args.signal,
            #args.tsites
        ],
        True
    )
    synced_vchroms = get_synced_chroms(
        args.vchroms,
        [
            args.sequence,
            args.average,
            #args.filters,
            #args.preferences,
            #args.validation,
            #args.vsites
        ],
        True
    )

    assert set(synced_tchroms).isdisjoint(set(synced_vchroms)), \
        "--tchroms and --vchroms shouldn't intersect. Exiting"

    synced_chroms = get_synced_chroms(  # call it just to take --chroms without possible regions
        args.chroms,
        [
            args.sequence,
            args.average,
        ],
        True
    )

    assert set(synced_tchroms).union(set(synced_vchroms)).issubset(set(synced_chroms)), \
        "--tchroms and --vchroms should be subset of --chroms. Exiting"

    synced_chroms = get_synced_chroms(
        set(synced_chroms) - set(synced_tchroms) - set(synced_vchroms),
        [
            args.sequence,
            args.average,
        ],
        True
    )

    synced_chroms.update(synced_tchroms)
    synced_chroms.update(synced_vchroms)

    assert len(synced_chroms) > 0, \
        "--chroms, --tchroms or --vchroms failed to sync with the provided files. Exiting"

    setattr(args, "tchroms", synced_tchroms)
    setattr(args, "vchroms", synced_vchroms)
    setattr(args, "chroms", synced_chroms)

    if args.threads is None:
        args.threads = 1  # TODO: maybe choose a smarter way to set default threads number


def assert_and_fix_args_for_normalization(args):
    synced_chroms = get_synced_chroms(
        args.chroms,
        [*args.signal, args.average],  # use * to unfold list of args.signal
        True
    )
    assert len(synced_chroms) > 0, \
        "Failed to sync chromosomes. Check --chroms. Exiting"
    setattr(args, "chroms", synced_chroms)

    if args.threads is None:
        args.threads = get_cpu_count()


def assert_and_fix_args_for_benchmarking(args):
    synced_chroms = get_synced_chroms(
        args.chroms,
        [args.prediction, args.control]
    )
    assert len(synced_chroms) > 0, "Failed to sync --chroms"
    setattr(args, "chroms", synced_chroms)  # now it's dict

    if args.threads is None:
        args.threads = get_cpu_count()


def assert_and_fix_args(args):

    args.loglevel = LOG_LEVELS[args.loglevel]
    if args.func == run_prediction:
        pass
    elif args.func == run_training:
        assert_and_fix_args_for_training(args)
    elif args.func == run_normalization:
        assert_and_fix_args_for_normalization(args)
    else:
        assert_and_fix_args_for_benchmarking(args)


def get_parser():
    # Parent (general) parser
    parent_parser = argparse.ArgumentParser(add_help=False)
    general_parser = argparse.ArgumentParser(description="maxATAC: \
        DeepCNN for motif binding prediction")
    subparsers = general_parser.add_subparsers()
    subparsers.required = True

    general_parser.add_argument(
        "--version", action="version", version=get_version(),
        help="Print version information and exit"
    )

    # Predict parser
    predict_parser = subparsers.add_parser(
        "predict",
        parents=[parent_parser],
        help="Run maxATAC prediction",
    )
    predict_parser.set_defaults(func=run_prediction)

    predict_parser.add_argument(
        "--models", dest="models", type=str, nargs="+",
        required=True,
        help="Trained model file(s)"
    )
    predict_parser.add_argument(
        "--quant", dest="quant", action='store_true',
        default=False,
        help="This argument should be set to true to build models based on quantitative data"
    )

    predict_parser.add_argument(
        "--train_tf", dest="train_tf", type=str,
        required=True,
        help="Transcription Factor to train on. Restricted to only 1 TF."
    )

    predict_parser.add_argument(
        "--test_cell_lines", dest="test_cell_lines", type=str, nargs="+",
        required=True,
        help="Cell lines for model testing. These cell lines will not be used in model training. cell lines must be delimited with , "
    )

    predict_parser.add_argument(
        "--average", dest="average", type=str,
        required=True,
        help="Average signal bigWig file"
    )

    predict_parser.add_argument(
        "--sequence", dest="sequence", type=str,
        required=True,
        help="Genome sequence 2bit file"
    )

    predict_parser.add_argument(
        "--meta_file", dest="meta_file", type=str,
        required=True,
        help="Meta file containing ATAC Signal and Bindings path for all cell lines (.tsv format)"
    )

    predict_parser.add_argument(
        "--minimum", dest="minimum", type=float,
        default=DEFAULT_MIN_PREDICTION,
        help="Minimum prediction value to be reported. Default: " + str(DEFAULT_MIN_PREDICTION)
    )

    predict_parser.add_argument(
        "--tmp", dest="tmp", type=str,
        default="./temp",
        help="Folder to save temporary data. Default: ./temp")

    predict_parser.add_argument(
        "--output", dest="output", type=str,
        default="./prediction_results",
        help="Folder for prediction results. Default: ./prediction_results")

    predict_parser.add_argument(
        "--keep", dest="keep", action="store_true",
        help="Keep temporary files. Default: False"
    )

    predict_parser.add_argument(
        "--threads", dest="threads", type=int,
        help="# of processes to run prediction in parallel. \
            Default: # of --models multiplied by # of --chroms"
    )

    predict_parser.add_argument(
        "--loglevel", dest="loglevel", type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL
    )
    
    predict_parser.add_argument(
        "--predict_roi", dest="predict_roi", type=str,
        help="Bed file with ranges for input sequences to be predicted. \
            Default: None, predictions are done on the whole chromosome length"
    )

    # Train parser
    train_parser = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        help="Run maxATAC training"
    )
    train_parser.set_defaults(func=run_training)

    train_parser.add_argument(
        "--sequence", dest="sequence", type=str,
        required=True,
        help="Genome sequence 2bit file"
    )
    
    train_parser.add_argument(
        "--quant", dest="quant", action='store_true',
        default=False,
        help="This argument should be set to true to build models based on quantitative data"
    )

    train_parser.add_argument(
        "--average", dest="average", type=str,
        required=True,
        help="Average signal bigWig file"
    )

    train_parser.add_argument(
        "--meta_file", dest="meta_file", type=str,
        required=True,
        help="Meta file containing ATAC Signal and Bindings path for all cell lines (.tsv format)"
    )

    train_parser.add_argument(
        "--train_roi", dest="train_roi", type=str,
        required=True,
        help="Bed file with ranges for input sequences. Required for peak-centric training of the model."
    )

    train_parser.add_argument(
        "--validate_roi", dest="validate_roi", type=str,
        required=True,
        help="Bed file  with ranges for input sequences to validate the model"
    )

    train_parser.add_argument(
        "--eval_roi", dest="eval_roi", type=str,
        required=True,
        help="Bed file  with ranges for input sequences to evaluate the model performance"
    )
    
    train_parser.add_argument(
        "--chroms", dest="chroms", type=str, nargs="+",
        required=True,
        help="Chromosome list for analysis. \
            Regions in a form of chrN:start-end are ignored. \
            Use --filters instead \
            Default: main human chromosomes, whole length"
    )
    
    train_parser.add_argument(
        "--tchroms", dest="tchroms", type=str, nargs="+",
        required=True,
        help="Chromosomes from --chroms fixed for training. \
            Regions in a form of chrN:start-end are ignored. \
            Use --filters instead \
            Default: None, whole length"
    )

    train_parser.add_argument(
        "--vchroms", dest="vchroms", type=str, nargs="+",
        required=True,
        help="Chromosomes from --chroms fixed for validation. \
            Regions in a form of chrN:start-end are ignored. \
            Use --filters instead \
            Default: None, whole length"
    )
    
    train_parser.add_argument(
        "--train_tf", dest="train_tf", type=str,
        required=True,
        help="Transcription Factor to train on. Restricted to only 1 TF."
    )

    train_parser.add_argument(
        "--arch", dest="arch", type=str,
        required=True,
        help="Specify the model architecture. Currently support DCNN_V2, RES_DCNN_V2, MM_DCNN_V2 and MM_Res_DCNN_V2"
    )

    train_parser.add_argument(
        "--test_cell_lines", dest="test_cell_lines", type=str, nargs="+",
        required=True,
        help="Cell lines for model testing. These cell lines will not be used in model training. cell lines must be delimited with , "
    )

    train_parser.add_argument(
        "--rand_ratio", dest="rand_ratio", type=float,
        required=True,
        help="Ratio for controlling fraction of random seqeuences in each traning batch. float [0, 1]"
    )


    train_parser.add_argument(
        "--seed", dest="seed", type=int,
        default=random.randint(1, 99999),
        help="Seed for pseudo-random generanor. Default: random int [1, 99999]"
    )

    train_parser.add_argument(
        "--weights", dest="weights", type=str,
        help="Weights to initialize model before training. Default: do not load"
    )

    train_parser.add_argument(
        "--epochs", dest="epochs", type=int,
        default=DEFAULT_TRAIN_EPOCHS,
        help="# of training epochs. Default: " + str(DEFAULT_TRAIN_EPOCHS)
    )

    train_parser.add_argument(
        "--batches", dest="batches", type=int,
        default=DEFAULT_TRAIN_BATCHES_PER_EPOCH,
        help="# of training batches per epoch. \
            Default: " + str(DEFAULT_TRAIN_BATCHES_PER_EPOCH)
    )

    train_parser.add_argument(
        "--batch_size", dest="batch_size", type=int,
        default=BATCH_SIZE,
        help="# of training batch size \
            Default: " + str(BATCH_SIZE)
    )

    train_parser.add_argument(
        "--val_batch_size", dest="val_batch_size", type=int,
        default=VAL_BATCH_SIZE,
        help="# of training validation batch size \
            Default: " + str(VAL_BATCH_SIZE)
    )

    train_parser.add_argument(
        "--lrate", dest="lrate", type=float,
        default=DEFAULT_ADAM_LEARNING_RATE,
        help="Learning rate. Default: " + str(DEFAULT_ADAM_LEARNING_RATE)
    )

    train_parser.add_argument(
        "--decay", dest="decay", type=float,
        default=DEFAULT_ADAM_DECAY,
        help="Learning rate decay. Default: " + str(DEFAULT_ADAM_DECAY)
    )

    train_parser.add_argument(
        "--prefix", dest="prefix", type=str,
        default="weights",
        help="Output prefix. Default: weights")

    train_parser.add_argument(
        "--output", dest="output", type=str,
        default="./training_results",
        help="Folder for training results. Default: ./training_results")

    train_parser.add_argument(
        "--plot", dest="plot", action="store_true",
        default=True,
        help="Plot model structure and training history. \
            Default: False"
    )

    train_parser.add_argument(
        "--threads", dest="threads", type=int,
        help="# of processes to run training in parallel. \
            Default: 1"
    )

    train_parser.add_argument(
        "--loglevel", dest="loglevel", type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL
    )

    # Normalize parser
    normalize_parser = subparsers.add_parser(
        "normalize",
        parents=[parent_parser],
        help="Run maxATAC normalization"
    )
    normalize_parser.set_defaults(func=run_normalization)

    normalize_parser.add_argument(
        "--signal", dest="signal", type=str, nargs="+",
        required=False,
        help="Input signal bigWig file(s) to be normalized by reference"
    )

    normalize_parser.add_argument(
        "--average", dest="average", type=str,
        required=True,
        help="Average signal bigWig file to be used as reference for normalization"
    )

    normalize_parser.add_argument(
        "--chroms", dest="chroms", type=str, nargs="+",
        default=DEFAULT_CHRS,
        help="Chromosomes list for analysis. \
            Regions in a form of chrN:start-end are ignored. \
            Default: main human chromosomes, whole length"
    )

    normalize_parser.add_argument(
        "--bin", dest="bin", type=int,
        default=DEFAULT_NORMALIZATION_BIN,
        help="Normalization bin size. \
            Default: " + str(DEFAULT_NORMALIZATION_BIN)
    )

    normalize_parser.add_argument(
        "--plot", dest="plot", action="store_true",
        help="Plot normalized signal(s) boxplots per chromosome. \
            Default: False"
    )

    normalize_parser.add_argument(
        "--prefix", dest="prefix", type=str,
        default="normalized",
        help="Output prefix. Default: normalized")

    normalize_parser.add_argument(
        "--output", dest="output", type=str,
        default="./normalization_results",
        help="Folder for normalization results. Default: ./normalization_results")

    normalize_parser.add_argument(
        "--threads", dest="threads", type=int,
        help="# of processes to run loading and exporing data in parallel. \
            Default: # of available CPUs minus 25 percent, minimum 1"
    )

    normalize_parser.add_argument(
        "--loglevel", dest="loglevel", type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=LOG_LEVELS.keys(),
        help="Logging level. Default: " + DEFAULT_LOG_LEVEL
    )

    # Benchmark parser
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        parents=[parent_parser],
        help="Run maxATAC benchmarking"
    )
    benchmark_parser.set_defaults(func=run_benchmarking)

    benchmark_parser.add_argument(
        "--prediction", dest="prediction", type=str,
        required=True,
        help="Prediction bigWig file"
    )

    benchmark_parser.add_argument(
        "--control", dest="control", type=str,
        required=True,
        help="Control bigWig file"
    )

    benchmark_parser.add_argument(
        "--chroms", dest="chroms", type=str, nargs="+",
        default=DEFAULT_CHRS,
        help="Chromosomes list for analysis. \
            Optionally with regions in a form of chrN:start-end. \
            Default: main human chromosomes, whole length"
    )

    benchmark_parser.add_argument(
        "--bin", dest="bin", type=int,
        default=DEFAULT_NORMALIZATION_BIN,
        help="Bin size to split prediction and control data before running prediction. \
            Default: " + str(DEFAULT_NORMALIZATION_BIN)
    )

    benchmark_parser.add_argument(
        "--plot", dest="plot", action="store_true",
        help="Plot PRC plot for every chromosome. \
            Default: False"
    )

    benchmark_parser.add_argument(
        "--output", dest="output", type=str,
        default="./benchmarking_results",
        help="Folder for benchmarking results. Default: ./benchmarking_results"
    )
    
    benchmark_parser.add_argument(
        "--threads", dest="threads", type=int,
        help="# of processes to run benchmarking in parallel. \
            Default: # of available CPUs minus 25 percent, min 1"
    )

    benchmark_parser.add_argument(
        "--loglevel", dest="loglevel", type=str,
        default=DEFAULT_LOG_LEVEL,
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
    
    if args.func == run_pc_training:
        args = normalize_args(
            args,
            [
                "func", "loglevel", "threads", "seed",
                "proportion", "vchroms", "tchroms",
                "chroms", "keep", "epochs", "batches",
                "prefix", "plot", "lrate", "decay", "bin",
                "minimum", "test_cell_lines", "rand_ratio", 
                "train_tf", "arch", "quant", "batch_size",
                "val_batch_size"
            ],
            cwd_abs_path
        )
    
    assert_and_fix_args(args)
    return args
