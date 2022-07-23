from logging import FATAL, ERROR, WARNING, INFO, DEBUG
import os

# Internal use

LOG_LEVELS = {
    "fatal": FATAL,
    "error": ERROR,
    "warning": WARNING,
    "info": INFO,
    "debug": DEBUG
}
LOG_FORMAT = "[%(asctime)s]\n%(message)s"
CPP_LOG_LEVEL = {
    FATAL: 3,
    ERROR: 3,
    WARNING: 2,
    INFO: 1,
    DEBUG: 0
}


# Paths to Genomic resource and scripts for preparing data. This is where most of the hardcoded paths are generated.
# If the user runs maxatac data the data will be installed here. The default arguments for some commands rely on these paths.
# The user can put the data anywhere, but they will need to adjust the paths for each file

def build_path_names(args):
    # import paths
    # logging.error(f"Generating Paths for genome build: {genome} \n")
    print(f"Generating Paths for genome build: {args.genome} \n")

    # build maxatac data path
    maxatac_data_path = os.path.join(os.path.expanduser('~'), "opt", "maxatac", "data")

    # build genome specific paths
    blacklist_path = os.path.join(maxatac_data_path,
                                  f"{args.genome}/{args.genome}_maxatac_blacklist.bed")  # maxATAC extended blacklist as bed
    blacklist_bigwig_path = os.path.join(maxatac_data_path,
                                         f"{args.genome}/{args.genome}_maxatac_blacklist.bw")  # maxATAC extended blacklist as bigwig
    chrom_sizes_path = os.path.join(maxatac_data_path, f"{args.genome}/{args.genome}.chrom.sizes")  # chrom sizes file
    sequence_path = os.path.join(maxatac_data_path, f"{args.genome}/{args.genome}.2bit")  # sequence 2bit

    # normalize paths
    DATA_PATH = os.path.normpath(maxatac_data_path)
    BLACKLISTED_REGIONS = os.path.normpath(blacklist_path)
    BLACKLISTED_REGIONS_BIGWIG = os.path.normpath(blacklist_bigwig_path)
    DEFAULT_CHROM_SIZES = os.path.normpath(chrom_sizes_path)
    REFERENCE_SEQUENCE_TWOBIT = os.path.normpath(sequence_path)


    return DATA_PATH, BLACKLISTED_REGIONS, BLACKLISTED_REGIONS_BIGWIG, DEFAULT_CHROM_SIZES, REFERENCE_SEQUENCE_TWOBIT
