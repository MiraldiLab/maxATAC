import pandas as pd
import os
from maxatac.utilities.peak_tools import call_peaks_per_chromosome
from maxatac.utilities.system_tools import get_dir
import logging
import multiprocessing
from multiprocessing import Pool, Manager
import pybedtools

def run_call_peaks(args):
    """Call peaks on a maxATAC signal track

    Args:
        input_bigwig: The path to the maxATAC bigwig file
        chromosomes: The list of chromosomes to call peaks for
        BIN_SIZE: The size of the bin to use for peak calling
        prefix: The prefix for the output filename
        threshold: The minimum threshold to use for peak calling
        OUT_DIR: The output directory to write the bed file
    
    Return:
        Write BED file
    """
    if args.prefix:
        prefix = args.prefix

    else:
        prefix = os.path.basename(args.input_bigwig).replace(".bw", "")

    output_dir = get_dir(args.OUT_DIR)

    results_filename = os.path.join(output_dir, prefix + "_" + str(args.BIN_SIZE) + "bp.bed")

    logging.error(f"Input filename: {args.input_bigwig}" +
                  f"Target chroms: {args.chromosomes}" +
                  f"Bin size: {args.BIN_SIZE}" +
                  f"Threshold for peak calling: {args.threshold}" +
                  f"\n Filename prefix: {prefix}" +
                  f"\n Output directory: {output_dir}" +
                  f"\n Output filename: {results_filename}")

    with Pool(int(multiprocessing.cpu_count()/2)) as p:
        results_list = p.starmap(call_peaks_per_chromosome,
                                [(args.input_bigwig, chromosome, args.threshold, args.BIN_SIZE) for chromosome in args.chromosomes]
                                )
        
    # Concatenate results lists into a dataframe
    results_df = pd.concat(results_list)

    # Convert the dataframe to a bedtools object
    peaks_bedtool = pybedtools.BedTool.from_dataframe(results_df)

    # Sort the bed intervals
    sorted_peaks = peaks_bedtool.sort()

    # Merge overlapping intervals
    merged_peaks = sorted_peaks.merge(c=4, o="max")

    # Convert bedtool object to dataframe
    BED_df = merged_peaks.to_dataframe()

    # Write dataframe to a bed format file
    BED_df.to_csv(results_filename, 
                sep="\t", 
                index=False, 
                header=False)
