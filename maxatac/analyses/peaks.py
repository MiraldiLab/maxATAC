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
        cutoff_type: Choose between Precision, Recall, log2FC, and F1 to choose your cutoffs
        cutoff_value: The value associated to cutoff type, i.e. Precision 0.75
        cutoff_file: Chr2 cutoff file found in /maxATAC/data/models/YOUR_TF_MODEL/YOUR_TF_MODEL_validationPerformance_vs_thresholdCalibration.tsv
        output: The output directory to write the bed file
    
    Return:
        Write BED file
    """
    if args.prefix:
        prefix = args.prefix

    else:
        prefix = os.path.basename(args.input_bigwig).replace(".bw", "")

    # call peaks on all chromosomes
    if args.chromosomes[0] == 'all':
        from maxatac.utilities.constants import AUTOSOMAL_CHRS as all_chr
        args.chromosomes = all_chr

    output_dir = get_dir(args.output)

    results_filename = os.path.join(output_dir, prefix + "_" + str(args.BIN_SIZE) + "bp.bed")

    # Find Threshold for specified cutoff values
    df = pd.read_csv(args.cutoff_file, sep='\t')

    # Get correct label
    dict = {"Precision": "Monotonic_Avg_Precision",
            "Recall": "Monotonic_Avg_Recall",
            "log2FC": "Monotonic_Avg_log2FC",
            "F1": "Avg_F1"
            }

    cutoff_type=dict[args.cutoff_type]

    if cutoff_type == "Avg_F1":

        # Find correct threshold for maximum F1 Score
        thresh = df.loc[df['Avg_F1'].idxmax()].Standard_Thresh
    else:
        cutoff_val=args.cutoff_value

        # Find correct threshold
        thresh = df.query(f"{cutoff_type} >= @cutoff_val").Standard_Thresh.tolist()[0]

    logging.error(f"Input filename: {args.input_bigwig}" +
                  f"\n Target chroms: {args.chromosomes}" +
                  f"\n Bin size: {args.BIN_SIZE}" +
                  f"\n Cutoff type for Threshold: {args.cutoff_type}" +
                  f"\n Cutoff value: {args.cutoff_value}" +
                  f"\n Corresponding Threshold for Cutoff Type and Value discovered: {thresh}" +
                  f"\n Filename prefix: {prefix}" +
                  f"\n Output directory: {output_dir}" +
                  f"\n Output filename: {results_filename}")

    with Pool(int(multiprocessing.cpu_count())) as p:
        results_list = p.starmap(call_peaks_per_chromosome,
                                [(args.input_bigwig, chromosome,thresh, args.BIN_SIZE) for chromosome in args.chromosomes]
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
