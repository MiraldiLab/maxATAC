import logging
import os
import pandas as pd

import pybedtools
from maxatac.utilities.system_tools import Mute, get_dir

with Mute():
    from maxatac.utilities.genome_tools import GenomicBins, RegionsOfInterest


def run_roi(args):
    """
    Generate the training and validation pools of genomic regions

    This method will use the run meta file to merge the ATAC-seq peaks and ChIP-seq peaks into BED files where each
    entry is an example region. ONE MAJOR ASSUMPTION IS THAT THE TEST CELL LINE IS NOT INCLUDED IN THE META FILE!!!!!

    The input meta file must have the columns in any order:

    TF | Cell_Line | ATAC_Signal_File | Binding_File | ATAC_Peaks | ChIP_peaks
    _________________
    Workflow Overview

    1) TrainingData: Import the ATAC-seq and ChIP-seq and filter for training chromosomes
    2) TrainingData.write_data: writes the ChIP, ATAC, and combined ROI pools with stats

    :param args: meta_file, output_dir, train_chroms, blacklist, training_prefix, validate_random_ratio,
    validate_chroms, chromosome_sizes, preferences, threads, validation_prefix

    :return: BED files for input into training with associated statistics
    """
    logging.error("Generating training regions of interest file: \n" +
                  "Meta Path: " + args.meta_file + "\n" +
                  "Output Directory: " + args.output + "\n" +
                  "Filename Prefix: " + args.prefix
                  )

    # Create the output directory set by the parser
    output_directory = get_dir(args.output)

    # Output filename for the bigwig predictions file based on the output directory and the prefix. Add the bw extension
    outfilename_ROI_bed = os.path.join(output_directory, args.prefix + ".bed")
    outfilename_bins_bed = os.path.join(output_directory, args.prefix + "_w" + str(args.window_size) + "_s" + str(
        args.step_size) + "_bins.bed")

    logging.error("Binning the genome with parameters: \n" +
                  "Bin chromosomes: \n   - " + "\n   - ".join(args.chromosomes) + "\n" +
                  "Window Size: " + str(args.window_size) + "\n" +
                  "Step Size: " + str(args.step_size)
                  )

    if args.bins:
        genomic_bins = pd.read_table(args.bins, header=None)

        genomic_bins.bins_bedtool = pybedtools.BedTool().from_dataframe(genomic_bins)

    else:
        # Bin the genome with bins equal to window size and shifted by step stize. Only use chromosomes of interest
        genomic_bins = GenomicBins(chromosome_sizes_file=args.chromosome_sizes,
                                   chromosomes=args.chromosomes,
                                   blacklist_path=args.blacklist,
                                   window_size=args.window_size,
                                   step_size=args.step_size)

        genomic_bins.bins_DF.to_csv(outfilename_bins_bed, header=False, index=False, sep="\t")

    logging.error("Importing ATAC-seq and ChIP-seq peaks")

    # Import the ATAC-seq and ChIP-seq peaks based on the meta file. Exclude blacklisted regions
    ROIPool = RegionsOfInterest(meta_path=args.meta_file,
                                blacklist=args.blacklist,
                                chromosomes=args.chromosomes)

    logging.error("Intersecting ATAC-seq and ChIP-seq peaks with genomic bins")

    # Intersect the genomic bins with the ATAC-seq and ChIP-seq peaks. This file will be parsed and written to a bed.
    binned_peaks = genomic_bins.bins_bedtool.intersect(ROIPool.combined_pool_bedtool, wa=True, wb=True)

    binned_peaks_df = pybedtools.BedTool.to_dataframe(binned_peaks)

    logging.error("Writing training ROI files")
    binned_peaks_df.columns = ["Chr", "Start", "Stop", "Chr_overlap", "Start_overlap",
                               "Stop_overlap", "ROI_Type", "Cell_Line"]

    binned_peaks_df.to_csv(outfilename_ROI_bed, sep="\t", header=False, index=False)

    logging.error("Done")
