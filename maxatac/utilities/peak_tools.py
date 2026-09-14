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
        
        logging.info(
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


CUTOFF_TYPES = ("Precision", "Recall", "F1")


def get_threshold(cutoff_file, cutoff_type, cutoff_val):
    """Look up the prediction-score threshold calibrated to a target metric value.

    Args:
        cutoff_file (str): Threshold calibration table written by `maxatac threshold`
            (`<prefix>_cross_celltype.tsv` or a per-sample `<sample>.tsv`), with columns
            Metric/Bin/Precision/Recall/Threshold/F1.
        cutoff_type (str): Which metric's bin grid to look the threshold up on.
            One of Precision, Recall, F1.
        cutoff_val (float): Target value on that metric's grid. Optional for F1, where
            omitting it selects the threshold with the highest F1.

    Returns:
        float: The threshold to apply to prediction scores.
    """
    df = pd.read_csv(cutoff_file, sep='\t')

    if "Metric" not in df.columns:
        raise ValueError(
            f"{cutoff_file} is not a threshold calibration table: no 'Metric' column. "
            "Legacy Standard_Thresh tables are no longer supported; regenerate the table "
            "with `maxatac threshold`."
        )

    if cutoff_type not in CUTOFF_TYPES:
        raise ValueError(f"Unknown cutoff type {cutoff_type}. Choose one of {', '.join(CUTOFF_TYPES)}.")

    rows = df[df["Metric"] == cutoff_type]

    if rows.empty:
        raise ValueError(f"{cutoff_file} has no {cutoff_type} rows to calibrate a threshold against.")

    if cutoff_type == "F1" and cutoff_val is None:
        selected = rows.loc[rows["F1"].idxmax()]

    else:
        if cutoff_val is None:
            raise ValueError(f"A cutoff value is required for cutoff type {cutoff_type}.")

        # Ceiling on the bin grid: the lowest bin that still meets the requested value.
        reachable = rows[rows["Bin"] >= cutoff_val].sort_values("Bin")

        if reachable.empty:
            raise ValueError(
                f"{cutoff_file} does not reach {cutoff_type} {cutoff_val}; "
                f"its highest {cutoff_type} bin is {rows['Bin'].max()}."
            )

        selected = reachable.iloc[0]

    logging.info(f"Threshold calibrated on {cutoff_type} bin {selected['Bin']}: {selected['Threshold']}" +
                 f"\n Achieved Precision: {selected['Precision']}" +
                 f"\n Achieved Recall: {selected['Recall']}" +
                 f"\n Achieved F1: {selected['F1']}")

    return float(selected["Threshold"])
