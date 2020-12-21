import logging
import numpy as np

from multiprocessing import Pool, Manager
from os import path

from maxatac.utilities.helpers import get_dir, get_rootname
from maxatac.utilities.bigwig import load_bigwig, dump_bigwig
from maxatac.utilities.plot import export_boxplot


def quantile_normalize(signal, reference):
    reference_sorted = np.sort(reference)
    return reference_sorted[signal.argsort().argsort()]


def load_chrom_signal_data(
    job_id,
    location,
    is_reference,
    chrom,  # (chrom_name, chrom_length)
    region,  # (start, end), we don't use it here, only print to log
    bin_size
):
    bin_count = int(chrom[1] / bin_size)  # need to floor the number

    logging.error(
        "Start loading job [" + str(job_id) + "]" +
        "\n  Target signal: " + location +
        "\n  Chromosome: " + \
             chrom[0] + ":" + \
             str(region[0]) + "-" + \
             str(region[1]) + " (" + \
             str(chrom[1]) + ")" +
        "\n  Binning: " + str(bin_count) + " bins * " + str(bin_size) + " bp" +
        "\n  Use it as reference: " + str(is_reference)
    )
    with load_bigwig(location) as data_stream:
        chrom_signal_data = np.nan_to_num(
            np.array(
                data_stream.stats(
                    chrom[0],
                    0,
                    chrom[1],
                    type="mean",
                    nBins=bin_count,
                    exact=True
                ),
                dtype=float  # need it to have NaN instead of None
            )
        )
    logging.error("Finished loading job " + str(job_id))
    return (location, chrom[0], chrom_signal_data, is_reference)


def get_scattered_loading_params(args):
    scattered_params = []
    job_id = 1
    locations = args.signal + [args.average]
    for i, location in enumerate(locations):
        for chrom_name, chrom_data in args.chroms.items():
            is_reference = i == len(locations) - 1  # the last one is average
            scattered_params.append(
                (
                    job_id,
                    location,
                    is_reference,
                    (chrom_name, chrom_data["length"]),
                    chrom_data["region"],
                    args.bin
                )
            )
            job_id += 1
    return scattered_params


def get_merged_data(raw_data, for_reference=False):
    raw_data_dict = {}  # refactore raw_data to dict for easy access

    for location, chrom_name, chrom_signal_data, is_reference in raw_data:
        if (not for_reference and is_reference) or (for_reference and not is_reference):
            continue
        raw_data_dict.setdefault(location, {})[chrom_name] = chrom_signal_data

    merged_data = []  # [(location, acc_signal)]
    for location, signal_per_chrom  in raw_data_dict.items():
        acc_signal = np.zeros(0)
        for chrom_name in sorted(signal_per_chrom.keys()):  # need it sorted
            acc_signal = np.append(
                acc_signal,
                signal_per_chrom[chrom_name]
            )
        merged_data.append((location, acc_signal))

    return merged_data


def get_normalized_signal(signal, reference_data):
    logging.error("Running normalization")
    normalized_signal = []
    for location, signal_data in signal:
        logging.error("  Processing " + get_rootname(location))
        normalized_signal.append(
            (
                location,
                quantile_normalize(signal_data, reference_data)
            )
        )
    return normalized_signal  # [(location, normalized_signal_data)]


def export_merged_data(merged_data_list, args):

    logging.error("Export normalized results")

    for location, merged_data in merged_data_list:
        results_filename = args.prefix + \
            "_" + get_rootname(location) + \
            "_ref_" + get_rootname(args.average) + \
            ".bigwig"

        results_location = path.join(get_dir(args.output), results_filename)

        chr_values = []
        chr_names = []

        with dump_bigwig(results_location) as data_stream:
            header = [(n, args.chroms[n]["length"]) for n in sorted(args.chroms.keys())]
            data_stream.addHeader(header)

            start = 0
            for chrom_name, chrom_length in header:
                bin_count = int(chrom_length / args.bin)
                end = start + bin_count
                data_per_chr = merged_data[start:end]
                chr_values.append(data_per_chr)
                chr_names.append(chrom_name)

                data_stream.addEntries(
                    chroms = [chrom_name] * data_per_chr.size,
                    starts = [i * args.bin for i in range(0, data_per_chr.size)],      # [0,  10, 20, 30, 40]
                    ends = [(i + 1) * args.bin for i in range(0, data_per_chr.size)],  # [10, 20, 30, 40, 50]
                    values = data_per_chr.tolist()
                )

                start = end

        if args.plot:
            export_boxplot(
                data=chr_values,
                file_location=results_location,
                names=chr_names,
                title="Quantile normalized " + get_rootname(location) + " by " + get_rootname(args.average)
            )
        
        logging.error("  Results are saved to: " + results_location)

    if args.plot:
        logging.error("Export boxplots for combined chromosomes")
        export_boxplot(
            data=[i[1] for i in merged_data_list],
            file_location=path.join(get_dir(args.output), "combined_chromosomes"),
            names=[get_rootname(i[0]) for i in merged_data_list],
            title="Quantile normalized merged chromosomes"
        )


def run_normalization(args):

    logging.error(
        "Normalization" +
        "\n  Target signal(s): \n   - " + "\n   - ".join(args.signal) +
        "\n  Average signal (reference distribution): " + args.average +
        "\n  Chromosomes: \n   - " + "\n   - ".join(
                str(k) + ":" + \
                str(v["region"][0])+ "-" + \
                str(v["region"][1])+ " (" + \
                str(v["length"])+ ")" for k, v in args.chroms.items()
            ) +
        "\n  Jobs count: " + 
        "\n     - loading: " + str(len(args.chroms) * (len(args.signal) + 1)) +
        "\n     - normalization: " + str(len(args.signal)) +
        "\n     - export: " + str(len(args.signal) + 1) +
        "\n  Threads count: " + str(args.threads) +
        "\n  Normalization bin size: " + str(args.bin) +
        "\n  Plot normalized signal(s) boxplots per chromosome: " + str(args.plot) +
        "\n  Logging level: " + logging.getLevelName(args.loglevel) +
        "\n  Output prefix: " + args.prefix + 
        "\n  Output directory: " + args.output
    )

    logging.error("Loading data using " + str(args.threads) + " threads")
    with Pool(args.threads) as p:
        raw_data = p.starmap(
            load_chrom_signal_data,
            get_scattered_loading_params(args)
        )

    logging.error(f"Preparing data for normalization. Merge chromosomes into genome-wide array")
    merged_signal_data = get_merged_data(raw_data=raw_data)  # [(location, signal_data)]
    merged_average_data = get_merged_data(raw_data=raw_data, for_reference=True)  # [(location, signal_data)]

    normalized_merged_signal_data = get_normalized_signal(  # [(location, normalized_signal_data)]
        signal=merged_signal_data,
        reference_data=merged_average_data[0][1])  # set average data directly

    export_merged_data(normalized_merged_signal_data + merged_average_data, args)
