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

# Genomic resource constants
blacklist_path = os.path.join(os.path.dirname(__file__), "../../data/hg38_composite_blacklist.bed")
blacklist_bigwig_path = os.path.join(os.path.dirname(__file__), "../../data/hg38_composite_blacklist.bw")
chrom_sizes_path = os.path.join(os.path.dirname(__file__), "../../data/hg38.chrom.sizes")
complement_path = os.path.join(os.path.dirname(__file__), "../../data/hg38_blacklist_complement_regions.bed")

BLACKLISTED_REGIONS = os.path.normpath(blacklist_path)
BLACKLISTED_REGIONS_BIGWIG = os.path.normpath(blacklist_bigwig_path)
DEFAULT_CHROM_SIZES = os.path.normpath(chrom_sizes_path)
COMPLEMENT_REGIONS = os.path.normpath(complement_path)

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
DEFAULT_CHRS = ["chr2", "chr3", "chr4", "chr5", "chr6",
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

DEFAULT_LOG_LEVEL = "error"

DEFAULT_TRAIN_EPOCHS = 20

DEFAULT_TRAIN_BATCHES_PER_EPOCH = 100

DEFAULT_NORMALIZATION_BIN = 100
DEFAULT_ADAM_LEARNING_RATE = 1e-3
DEFAULT_ADAM_DECAY = 1e-5
DEFAULT_VALIDATE_RAND_RATIO = .7

# Can be changed without problems
BATCH_SIZE = 1000
VAL_BATCH_SIZE = 1000


CHR_POOL_SIZE = 1000
FLANK_LENGTH = 100  # make sure that 2 * FLANK_LENGTH < INPUT_LENGTH
BP_ORDER = ["A", "C", "G", "T"]
PHASES = [0, 0.5]  # each item should belong to [0, 1)
INPUT_FILTERS = 15
INPUT_KERNEL_SIZE = 7
INPUT_LENGTH = 1024
OUTPUT_LENGTH = 32 # INPUT_LENGTH/BP_RESOLUTION
INPUT_ACTIVATION = "relu"
KERNEL_INITIALIZER = "glorot_uniform" #use he_normal initializer if activation is RELU
PADDING = "same"
FILTERS_SCALING_FACTOR = 1.5
PURE_CONV_LAYERS = 4
CONV_BLOCKS = 6
DNA_INPUT_CHANNELS = 4
ATAC_INPUT_CHANNELS = 2
DILATION_RATE = [1, 1, 2, 4, 8, 16]
BP_RESOLUTION = 32
OUTPUT_FILTERS = 1
OUTPUT_KERNEL_SIZE = 1
#BINARY_OUTPUT_ACTIVATION = "sigmoid"
#QUANT_OUTPUT_ACTIVATION = "linear"
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

# Factor for scaling Targets for quant models.
QUANT_TARGET_SCALE_FACTOR = 10
# I wouldn't recommend to change without looking into code

INPUT_CHANNELS = 5
TRAIN_MONITOR = "val_loss"
