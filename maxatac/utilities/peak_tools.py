import pandas as pd
import numpy as np
from maxatac.utilities.genome_tools import load_bigwig
import logging

def call_peaks_per_chromosome(bigwig_path, chrom_name, threshold, bin_size=200):
    """Call peaks on a maxATAC prediction signal track

    Args:
        signal_stream (str): Loaded bigwig signal stream.
        chrom_name (str): Name of the chromosome of interest.
        threshold (float): The threshold value to use to call peaks.
        bin_size (int, optional): The size of the bins to use in base pairs. Defaults to 200.

    Returns:
        Dataframe: A dataframe of genomic regions that above the given threshold
        
    Example:
    bed_regions_df = call_peaks(signal_stream, "chr19", .75)
    """
    with load_bigwig(bigwig_path) as signal_stream:
        # Get the chromosome lengths
        chrom_length = signal_stream.chroms(chrom_name)
        
        # Get the number of bins per chromosome
        bin_count = int(int(chrom_length) / int(bin_size))
        
        logging.error(
        "Start loading chromosome " + chrom_name +
        "\n  Input signal: " + bigwig_path +
        "\n  Binning: " + str(bin_count) + " bins * " + str(bin_size) + " bp"
        )
        
        # Get the chromosome valies into an np array.
        chrom_vals = np.nan_to_num(np.array(signal_stream.stats(chrom_name,
                                                                0,
                                                                chrom_length,
                                                                type="max",
                                                                nBins=bin_count,
                                                                exact=True),
                                            dtype=float  # need it to have NaN instead of None
                                            ))
        
        # Find threshold around the given recall or precision
        target_bin_idx_list = np.argwhere(chrom_vals >= threshold)
                
        # Create an empty list to hold the results
        BIN_list = []

        # Loop over bin list and convert to genomic regions
        for prediction_bin in target_bin_idx_list:
            
            start = prediction_bin * bin_size
            
            BIN_list.append([chrom_name, 
                            start[0], 
                            start[0] + bin_size + 1, 
                            chrom_vals[prediction_bin][0]
                            ])

    return pd.DataFrame(BIN_list, columns=["chr", "start", "end", "score"])
