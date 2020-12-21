import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool
from os import path
from sklearn.metrics import (
    precision_recall_curve,
    plot_precision_recall_curve
)

from maxatac.utilities.helpers import get_dir, get_rootname, load_bigwig
from maxatac.utilities.plot import export_prc


def run_chrom_benchmarking(
    job_id,
    prediction,
    control,
    bin_size,
    chrom,  # (chrom_name, chrom_length)
    plot,
    output
):
    results_filename = get_rootname(prediction) + \
        "_" + get_rootname(control) + \
        "_" + chrom[0] + \
        "_" + str(bin_size) + \
        ".tsv"
    results_location = path.join(get_dir(output), results_filename)

    bin_count = int(chrom[1] / bin_size)  # need to floor the number
    logging.error(
        "Start job [" + str(job_id) + "]" +
        "\n  Prediction file:" + prediction +
        "\n  Control file: " + control +
        "\n  Chromosome: " + chrom[0] + " (" + str(chrom[1]) + ")" +
        "\n  Binning: " + str(bin_count) + " bins * " + str(bin_size) + " bp"
        "\n  Results location: " + results_location
    )
    
    logging.error("  Loading data")
    with \
            load_bigwig(prediction) as prediction_stream, \
            load_bigwig(control) as control_stream:
        prediction_chrom_data = np.nan_to_num(
            np.array(
                prediction_stream.stats(
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
        control_chrom_data = np.nan_to_num(
            np.array(
                control_stream.stats(
                    chrom[0],
                    0,
                    chrom[1],
                    type="mean",
                    nBins=bin_count,
                    exact=True
                ),
                dtype=float  # need it to have NaN instead of None
            )
        ) > 0  # to convert to boolean array
        
        logging.error("  Running PRC")
        precision, recall, thresholds = precision_recall_curve(
            control_chrom_data,
            prediction_chrom_data
        )
        prc_data = pd.DataFrame({"recall":recall, "precision":precision}) 
        prc_data.to_csv(results_location, index=False, sep="\t")

        if plot:
            logging.error("  Export PRC plot")        
            export_prc(
                precision=precision,
                recall=recall,
                file_location=results_location
            )
            
        logging.error(
            "Finish job [" + str(job_id) + "]" +
            "\n  Results are saved to: " + results_location
        )


def get_scattered_params(args):
    scattered_params = []
    job_id = 1
    for chrom_name, chrom_length in args.chroms.items():
        scattered_params.append(
            (
                job_id,
                args.prediction,
                args.control,
                args.bin,
                (chrom_name, chrom_length),
                args.plot,
                args.output
            )
        )
        job_id += 1
    return scattered_params


def run_benchmarking(args):
    logging.error(
        "Benchmarking" +
        "\n  Prediction file:" + args.prediction +
        "\n  Control file: " + args.control +
        "\n  Bin size: " + str(args.bin) +
        "\n  All chromosomes: " + ", ".join(args.chroms) +
        "\n  Jobs count: " + str(len(args.chroms)) +
        "\n  Parallelization: " + str(args.threads) +
        "\n  Export PRC plot: " + str(args.plot) +
        "\n  Logging level: " + logging.getLevelName(args.loglevel) +
        "\n  Output directory: " + args.output
    )

    with Pool(args.threads) as p:
        raw_data = p.starmap(
            run_chrom_benchmarking,
            get_scattered_params(args)
        )