import pyBigWig

from maxatac.utilities.system_tools import Mute

with Mute():
    from maxatac.utilities.genome_tools import load_bigwig, build_chrom_sizes_dict, get_bigwig_values
    import numpy as np


def run_max_combine(args):
    chromosome_sizes_dictionary = build_chrom_sizes_dict([args.chromosome], args.chrom_sizes)

    chrom_length = chromosome_sizes_dictionary[args.chromosome]

    print("Import the average signal")
    average_signal_data = get_bigwig_values(args.average_signal,
                                            args.chromosome,
                                            chrom_length)

    print("Import the prediction signal")
    maxatac_prediction_data = get_bigwig_values(args.maxatac_prediction,
                                                args.chromosome,
                                                chrom_length)

    mean_combined_array = (average_signal_data + maxatac_prediction_data)/2

    with pyBigWig.open(args.output, "w") as output_bw:
        # Add a header based on the chromosomes in the chromosome sizes dictionary
        header = [(args.chromosome, chrom_length)]

        output_bw.addHeader(header)

        chrom_vals = np.zeros(chrom_length)

        # Write the entries to the bigwig for this chromosome. Current resolution is at 1 bp.
        output_bw.addEntries(chroms=args.chromosome,
                             starts=0,
                             ends=chrom_length,
                             span=1,
                             step=1,
                             values=mean_combined_array.tolist()
                             )
