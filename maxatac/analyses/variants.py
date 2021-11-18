import logging
import os
import timeit
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.prediction_tools import make_stranded_predictions, write_predictions_to_bigwig, create_prediction_regions
from maxatac.utilities.constants import INPUT_LENGTH, DEFAULT_CHROM_SIZES, BLACKLISTED_REGIONS
from maxatac.utilities.variant_tools import import_bed, intersect_bins_targets
from maxatac.utilities.genome_tools import build_chrom_sizes_dict, filter_chrom_sizes
import numpy as np

def run_variants(args):
    """Predict TF binding using a variant call format file to make sequence specific predictions
    
    The most basic function for this module is to predict in a single region given a single individuals variant calls.

    Args:
        args ([type]): [description]
    """
    # Start Timer
    startTime = timeit.default_timer()

    # Create the output directory set by the parser
    output_directory = get_dir(args.output)

    # Output filename for the bigwig predictions file based on the output directory and the prefix. Add the bw extension
    outfile_name_bigwig = os.path.join(output_directory, args.prefix + ".bw")

    filtered_chrom_sizes_file = os.path.join(output_directory, "filtered_chrom_sizes.txt")
        
    logging.error("Prediction Parameters \n" +
                  "Output filename: " + outfile_name_bigwig + "\n" +
                  "Target signal: " + args.signal + "\n" +
                  "Sequence data: " + args.sequence + "\n" +
                  "Threads count: " + str(args.threads) + "\n" +
                  "Output directory: " + str(output_directory) + "\n" +
                  "Batch Size: " + str(args.batch_size) + "\n" +
                  "Output filename: " + outfile_name_bigwig + "\n"
                  )
    
    logging.error("Import target regions BED file")
    
    regions_pool = import_bed(bed_file=args.roi,
                              blacklist=args.blacklist)

    chromosome_list = np.sort(regions_pool["chr"].unique()).tolist()

    logging.error("Create prediction regions")
    # Bedtools windowmake used in create_prediction_regions expects a text file of chrom sizes. 
    # To speed up the process, we create a chrom.sizes file specific for your chromosome list.
    # This will save a file that will be used instead of the full chrom sizes file
    target_chrom_sizes = filter_chrom_sizes(DEFAULT_CHROM_SIZES, 
                       chromosomes=chromosome_list,
                       target_chrom_sizes_file=filtered_chrom_sizes_file)
    
    chrom_sizes_dict = build_chrom_sizes_dict(chromosome_list=chromosome_list, chrom_sizes_filename=filtered_chrom_sizes_file)

    windowed_genome = create_prediction_regions(region_length=INPUT_LENGTH,
                                                chromosomes=chromosome_list,
                                                chrom_sizes=target_chrom_sizes,
                                                blacklist=args.blacklist,
                                                step_size=256)

    prediction_regions = intersect_bins_targets(windowed_genome, regions_pool, blacklist=args.blacklist)
    
    logging.error("Make forward strand predictions")

    forward_strand_predictions = make_stranded_predictions(signal=args.signal,
                                                        sequence=args.sequence,
                                                        models=args.models,
                                                        predict_roi_df=prediction_regions,
                                                        batch_size=args.batch_size,
                                                        use_complement=False)
    
    logging.error("Write predictions to a bigwig file")

    # Write the predictions to a bigwig file
    write_predictions_to_bigwig(forward_strand_predictions,
                                chrom_sizes_dictionary=chrom_sizes_dict,
                                chromosomes=chromosome_list,
                                output_filename=os.path.join(output_directory, args.prefix + ".bw"))

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
args = Namespace(output="/Users/caz3so/scratch/maxatac_test/variants/output", 
                 prefix='AD4_maxATAC',
                 signal="/Users/caz3so/scratch/20211012_maxATAC_atopicDerm_samples/outputs/minmax_bigwig/AD4__minmax01_percentile99.bw",
                 sequence="/Users/caz3so/scratch/maxATAC/data/genome_inf/hg38/hg38.2bit",
                 models="/Users/caz3so/scratch/maxatac_test/variants/input/ELK1_binary_revcomp99_fullModel_RR0_95.h5",
                 threads=3,
                 chromosomes=["chr1", "chr14"],
                 chromosome_sizes=DEFAULT_CHROM_SIZES,
                 batch_size=1,
                 blacklist=BLACKLISTED_REGIONS,
                 roi="/Users/caz3so/scratch/20211012_maxATAC_atopicDerm_samples/data/Eapen_2021_atopicDermatitis_atac_variant_overlap_hglft_variantFound.bed")

run_variants(args)