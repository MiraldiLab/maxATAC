import logging
import numpy as np
import pyBigWig
import os
import tqdm

from maxatac.utilities.genome_tools import build_chrom_sizes_dict, FindGenomicMinMax, MinMaxNormalizeArray
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.constants import DEFAULT_CHRS


def run_normalization(args):
    basename = os.path.basename(args.signal).split(".bw")[0]

    OUTPUT_FILENAME = os.path.join(args.output, basename + "_minmax01.bw")

    output_dir = get_dir(args.output)

    logging.error("Normalization" +
                  "\n  Input bigwig file: " + args.signal +
                  "\n  Output filename: " + OUTPUT_FILENAME +
                  "\n  Output directory: " + output_dir
                  )

    chromosome_length_dictionary = build_chrom_sizes_dict(DEFAULT_CHRS, args.chrom_sizes)

    genome_min, genome_max = FindGenomicMinMax(args.signal)

    logging.error("Normalize and Write BigWig file")

    with pyBigWig.open(args.signal) as input_bw, pyBigWig.open(OUTPUT_FILENAME, "w") as output_bw:
        header = [(x, chromosome_length_dictionary[x]) for x in sorted(DEFAULT_CHRS)]

        output_bw.addHeader(header)

        # Create a status bar for to look fancy and count what chromosome you are on
        chrom_status_bar = tqdm.tqdm(total=len(DEFAULT_CHRS), desc='Chromosomes Processed', position=0)

        for chrom_name, chrom_length in header:
            chr_vals = np.nan_to_num(input_bw.values(chrom_name, 0, chrom_length, numpy=True))

            normalized_signal = MinMaxNormalizeArray(chr_vals, genome_min, genome_max)

            output_bw.addEntries(
                chroms=chrom_name,
                starts=0,
                ends=chrom_length,
                span=1,
                step=1,
                values=normalized_signal.tolist()
            )

            chrom_status_bar.update(1)

    logging.error("Results saved to: " + output_dir)
