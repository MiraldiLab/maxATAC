from logging import FATAL, ERROR, WARNING, INFO, DEBUG
import os

# Parser Constants
LOG_LEVELS = {
    "fatal": FATAL,
    "error": ERROR,
    "warning": WARNING,
    "info": INFO,
    "debug": DEBUG
}

DEFAULT_LOG_LEVEL = "error"

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
chrom_sizes_path = os.path.join(os.path.dirname(__file__), "../../data/hg38.chrom.sizes")

BLACKLISTED_REGIONS = os.path.normpath(blacklist_path)
DEFAULT_CHROM_SIZES= os.path.normpath(chrom_sizes_path)

# Default chromosome sets
AUTOSOMAL_CHRS = [
    "chr1",  "chr2",  "chr3",  "chr4",  "chr5",  "chr6",
    "chr7",  "chr8",  "chr9",  "chr10", "chr11", "chr12",
    "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
    "chr19", "chr20", "chr21", "chr22"
]

DEFAULT_CHRS = [
    "chr1",  "chr2",  "chr3",  "chr4",  "chr5",  "chr6",
    "chr7",  "chr8",  "chr9",  "chr10", "chr11", "chr12",
    "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
    "chr19", "chr20", "chr21", "chr22", "chrX"
]

DEFAULT_TRAIN_CHRS = ["chr3",  "chr4",  "chr5",  "chr6",
    "chr7", "chr9",  "chr10", "chr11", "chr12",
    "chr13", "chr14", "chr15", "chr16", "chr17", 
    "chr18", "chr20", "chr21", "chr22"]

DEFAULT_VALIDATE_CHRS = ["chr2", "chr19"]

DEFAULT_TEST_CHRS = ["chr1", "chr8"]

# Training  Constants
DEFAULT_TRAIN_EPOCHS = 20
DEFAULT_TRAIN_BATCHES_PER_EPOCH = 1000
DEFAULT_VALIDATE_BATCHES_PER_EPOCH = 1000
DEFAULT_VALIDATE_BATCH_SIZE = 1000
DEFAULT_TRAIN_BATCH_SIZE = 1000
DEFAULT_VALIDATE_RAND_RATIO = .5
DEFAULT_TRAIN_RAND_RATIO = .5

# Model Constants
DEFAULT_ADAM_LEARNING_RATE = 1e-3
DEFAULT_ADAM_DECAY = 1e-5
BP_ORDER = ["A", "C", "G", "T"]
PHASES = [0, 0.5]  # each item should belong to [0, 1)
INPUT_FILTERS = 15
INPUT_KERNEL_SIZE = 7
INPUT_LENGTH = 1024
FLANK_LENGTH = 100  # make sure that 2 * FLANK_LENGTH < INPUT_LENGTH
OUTPUT_LENGTH = 32 # INPUT_LENGTH/BP_RESOLUTION
INPUT_ACTIVATION = "relu"
PADDING = "same"
FILTERS_SCALING_FACTOR = 1.5
CONV_BLOCKS = 6
DILATION_RATE = [1, 1, 2, 4, 8, 16]
BP_RESOLUTION = 32
OUTPUT_FILTERS = 1
OUTPUT_KERNEL_SIZE = 1
OUTPUT_ACTIVATION = "sigmoid"
POOL_SIZE = 2
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
TRAIN_SCALE_SIGNAL = (0.9, 1.15)  # min max scaling ranges
INPUT_CHANNELS = 6
TRAIN_MONITOR = "val_loss"

# Random Regions Generator Constants
CHR_POOL_SIZE = 10000

# Prediction Constants
DEFAULT_MIN_PREDICTION = 0.001  # min prediction value to be reported in the output
MIN_PREDICTION = 0.01  # min prediction value to report in output
DEFAULT_ROUND=6
DEFAULT_PREDICTION_BATCH_SIZE=1000

# Benchmarking Constants
DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION="max"
DEFAULT_BENCHMARKING_BIN_SIZE = 32
