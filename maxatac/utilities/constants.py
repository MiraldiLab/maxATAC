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

# build maxatac data path
maxatac_data_path = os.path.join(os.path.expanduser('~'), "opt", "maxatac", "data")

# build processing script path names
prepare_atac_script_dir = os.path.join(maxatac_data_path, "scripts", "ATAC",
                                       "ATAC_bowtie2_pipeline.sh")  # bulk processing script
prepare_scatac_script_dir = os.path.join(maxatac_data_path, "scripts", "ATAC",
                                         "scatac_generate_bigwig.sh")  # scatac processing script

PREPARE_BULK_SCRIPT = os.path.normpath(prepare_atac_script_dir)
PREPARE_scATAC_SCRIPT = os.path.normpath(prepare_scatac_script_dir)

cisbp_motifs_meta = os.path.join(maxatac_data_path, "motifs", "CISBP_v2_motif_mapping_noTransfac.txt")
zipped_motifs_dir = os.path.join(maxatac_data_path, "motifs", "all_motifs.tar.gz")
motifs_dir = os.path.join(maxatac_data_path, "motifs", "all_motifs")

# Default chromosome sets
ALL_CHRS = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6",
            "chr7", "chr8", "chr9", "chr10", "chr11", "chr12",
            "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
            "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"
            ]

AUTOSOMAL_CHRS = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6",
                  "chr7", "chr8", "chr9", "chr10", "chr11", "chr12",
                  "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
                  "chr19", "chr20", "chr21", "chr22"
                  ]

# Defualt chrs excludes 1,8
DEFAULT_TRAIN_VALIDATE_CHRS = ["chr2", "chr3", "chr4", "chr5", "chr6",
                               "chr7", "chr9", "chr10", "chr11", "chr12",
                               "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
                               "chr19", "chr20", "chr21", "chr22", "chrX"
                               ]

# Default train chrs exclude 1,2,8,19,X,Y,M
DEFAULT_TRAIN_CHRS = ["chr3", "chr4", "chr5", "chr6",
                      "chr7", "chr9", "chr10", "chr11", "chr12",
                      "chr13", "chr14", "chr15", "chr16", "chr17",
                      "chr18", "chr20", "chr21", "chr22"]

DEFAULT_VALIDATE_CHRS = ["chr2", "chr19"]

DEFAULT_TEST_CHRS = ["chr1", "chr8"]

DEFAULT_LOG_LEVEL = "info"

DEFAULT_TRAIN_EPOCHS = 20

DEFAULT_TRAIN_BATCHES_PER_EPOCH = 100

DEFAULT_ADAM_LEARNING_RATE = 1e-3
DEFAULT_ADAM_DECAY = 1e-5
DEFAULT_VALIDATE_RAND_RATIO = .7

# Can be changed without problems
BATCH_SIZE = 1000
VAL_BATCH_SIZE = 1000
BP_DICT = {"A":0, "C":1, "G":2, "T":3}
CHR_POOL_SIZE = 1000
BP_ORDER = ["A", "C", "G", "T"]
INPUT_FILTERS = 15
INPUT_KERNEL_SIZE = 7
INPUT_LENGTH = 1024
OUTPUT_LENGTH = 32  # INPUT_LENGTH/BP_RESOLUTION
INPUT_ACTIVATION = "relu"
KERNEL_INITIALIZER = "glorot_uniform"  # use he_normal initializer if activation is RELU
PADDING = "same"
FILTERS_SCALING_FACTOR = 1.5
PURE_CONV_LAYERS = 4
CONV_BLOCKS = 6
DNA_INPUT_CHANNELS = 4
DILATION_RATE = [1, 1, 2, 4, 8, 16]
BP_RESOLUTION = 32
OUTPUT_FILTERS = 1
OUTPUT_KERNEL_SIZE = 1
POOL_SIZE = 2
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
TRAIN_SCALE_SIGNAL = (0.9, 1.15)  # min max scaling ranges

# Prediction Constants
DEFAULT_MIN_PREDICTION = 0.001  # min prediction value to be reported in the output
DEFAULT_ROUND = 9
DEFAULT_PREDICTION_BATCH_SIZE = 10000
OUTPUT_ACTIVATION = "sigmoid"

# Benchmarking Constants
DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION = "max"
DEFAULT_BENCHMARKING_BIN_SIZE = 32

INPUT_CHANNELS = 5
TRAIN_MONITOR = "val_loss"
