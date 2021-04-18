from maxatac.utilities.genome_tools import build_chrom_sizes_dict, load_bigwig
import numpy as np
import pandas as pd
import os
from maxatac.utilities.peak_tools import get_genomic_locations
from maxatac.utilities.system_tools import get_dir


def call_peaks(args):
    interval_list = []
    output_dir = get_dir(args.output_dir)

    results_filename = os.path.join(output_dir,
                                    args.prefix + "_" + str(args.bin_size) + "bp.bed")

    # Loop through the chromosomes and average the values across files
    with load_bigwig(args.input_bigwig) as signal_stream:
        chrom_dict = signal_stream.chroms()
        for chrom_name, chrom_length in chrom_dict.items():
            bin_count = int(int(chrom_length) / int(args.bin_size))  # need to floor the number

            chrom_vals = np.nan_to_num(np.array(signal_stream.stats(chrom_name,
                                                                    0,
                                                                    chrom_length,
                                                                    type="max",
                                                                    nBins=bin_count,
                                                                    exact=True),
                                                dtype=float  # need it to have NaN instead of None
                                                ))

            interval_list.append(get_genomic_locations(chrom_vals, args.threshold, chrom_name, args.bin_size))

    BED_DF = pd.concat(interval_list)

    BED_DF.to_csv(results_filename, sep="\t", index=False, header=False)