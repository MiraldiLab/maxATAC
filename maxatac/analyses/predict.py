"""
Predict TF binding with a maxATAC model
"""
import logging
import os
import glob
import timeit
import pandas as pd
import multiprocessing
import pybedtools
from multiprocessing import Pool, Manager
from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict
    from maxatac.utilities.peak_tools import get_threshold
    from maxatac.utilities.prediction_tools import write_predictions_to_bigwig, \
        create_prediction_regions, make_stranded_predictions


def run_prediction(args):
    """
    Predict TF binding with a maxATAC model. The user can provide a bed file of regions to predict on,
    called windows, or prediction regions can be created based on the chromosome of interest. The default prediction
    will predict across all autosomal chromosomes.

    Peak file requirements for prediction. You must have at least a 3 column file with chromosome, start,
    and stop coordinates.

    Windows file requirements for prediction. These windows will be directly input into the maxATAC data generator
    and should be 1,024 bp wide. The window step should be uniform across the chromosome.

    The user can decide whether to make only predictions on the forward strand or also make prediction on the reverse
    strand. If the user wants both strand, signal tracks will be produced for the forward, reverse, and mean-combined
    bigwig signal tracks will be produced.

    Workflow overview

    1) Create directories and set up filenames
    2) Prepare regions for prediction. Either import user defined regions or create regions based on chromosomes list.
    3) Make predictions.
    4) Convert predictions to bigwig format and write results.
    5) Write predictions to an optional BED formated file of regions above a specific threshold.

    Args: TF, output_directory, name, signal, sequence, model, threads, batch_size, roi, cutoff_type, cutoff_value,
    cutoff_file, chrom_sizes, blacklist, average, windows, loglevel, step_size, chromosomes, skip_call_peaks
    """
    # Start Timer
    startTime = timeit.default_timer()

    # If the user provides the TF name,
    if args.TF:
        args.model = glob.glob(os.path.join(args.DATA_PATH, "models", args.TF, args.TF + "*.h5"))[0]

        args.cutoff_file = glob.glob(os.path.join(args.DATA_PATH, "models", args.TF, args.TF + "*.tsv"))[0]

    else:
        pass

    logging.info(f"Using maxATAC model: {args.model} to make predictions")

    # predict on all chromosomes
    if args.chromosomes[0] == 'all':
        from maxatac.utilities.constants import AUTOSOMAL_CHRS as all_chr
        args.chromosomes = all_chr

    # Create the output directory set by the parser
    logging.debug(f"Make output directory: {args.output_directory}")
    output_directory = get_dir(args.output_directory)

    # Output filename for the bigwig predictions file based on the output directory and the prefix. Add the bw extension
    outfile_name_bigwig = os.path.join(output_directory, args.name + ".bw")

    # The function build_chrom_sizes_dict is used to make sure regions fall within chromosome bounds.
    # Create a dictionary of chromosome sizes used to make the bigwig files
    chrom_sizes_dict = build_chrom_sizes_dict(args.chromosomes, args.chrom_sizes)
    logging.debug(f"Chromosome size dictionary: {chrom_sizes_dict}")

    # Import the regions for prediction.
    logging.info("Create prediction regions")
    regions_pool = create_prediction_regions(chromosomes=args.chromosomes,
                                             chrom_sizes=args.chrom_sizes,
                                             blacklist=args.blacklist,
                                             step_size=args.step_size,
                                             peaks=args.roi,
                                             windows=args.windows
                                             )

    # Find the chromosomes for which we can make predictions based on the requested chroms
    # and the BED regions provided in the ROI file
    chrom_list = list(set(args.chromosomes).intersection(set(regions_pool["chr"].unique())))

    logging.info("Prediction Parameters \n" +
                 f"Output filename: {outfile_name_bigwig} \n" +
                 f"Target signal: {args.signal} \n" +
                 f"Sequence data: {args.sequence} \n" +
                 f"Model: {args.model} \n" +
                 "Chromosome requested: \n   - " + "\n    -".join(args.chromosomes) + "\n" +
                 "Chromosomes in final prediction set: \n   - " + "\n    -".join(chrom_list) + "\n" +
                 f"Output directory: {output_directory} \n" +
                 f"Batch Size: {args.batch_size} \n" +
                 f"Output filename: {outfile_name_bigwig}"
                 )

    with Pool(int(multiprocessing.cpu_count())) as p:
        forward_strand_predictions = p.starmap(make_stranded_predictions,
                                               [(regions_pool,
                                                 args.signal,
                                                 args.sequence,
                                                 args.model,
                                                 args.batch_size,
                                                 False,
                                                 chromosome) for chromosome in chrom_list])

    # Write the predictions to a bigwig file and add name to args
    prediction_bedgraph = pd.concat(forward_strand_predictions)

    logging.info("Write predictions to a bigwig file")
    write_predictions_to_bigwig(prediction_bedgraph,
                                output_filename=outfile_name_bigwig,
                                chrom_sizes_dictionary=chrom_sizes_dict,
                                chromosomes=chrom_list
                                )

    # If a cutoff file is provided, call peaks
    if args.cutoff_file and args.skip_call_peaks is False:
        args.input_bigwig = outfile_name_bigwig

        peaks_filename = os.path.join(output_directory, args.name + "_peaks.bed")

        thresh = get_threshold(cutoff_file=args.cutoff_file,
                               cutoff_type=args.cutoff_type,
                               cutoff_val=args.cutoff_value)

        logging.info(f"Writing predictions to a BED file: {peaks_filename}" +
                     f"\n Cutoff type for Threshold: {args.cutoff_type}" +
                     f"\n Cutoff value: {args.cutoff_value}" +
                     f"\n Corresponding Threshold for Cutoff Type and Value discovered: {thresh}" +
                     f"\n Output filename: {peaks_filename}")

        logging.debug(f"Filtering predictions by threshold score.")
        prediction_bedgraph = prediction_bedgraph[prediction_bedgraph["score"] >= thresh]

        # Convert the dataframe to a bedtools object
        peaks_bedtool = pybedtools.BedTool.from_dataframe(prediction_bedgraph)

        # Sort the bed intervals
        sorted_peaks = peaks_bedtool.sort()

        # Merge overlapping intervals
        merged_peaks = sorted_peaks.merge(c=4, o="max")

        # Convert bedtool object to dataframe
        prediction_bed = merged_peaks.to_dataframe()

        logging.debug(f"Writing BED file.")
        # Write dataframe to a bed format file
        prediction_bed.to_csv(peaks_filename,
                  sep="\t",
                  index=False,
                  header=False)


    # Measure end time of training
    stopTime = timeit.default_timer()
    totalTime = stopTime - startTime

    # Output running time in a nice format.
    mins, secs = divmod(totalTime, 60)
    hours, mins = divmod(mins, 60)

    logging.info("Total Prediction time: %d:%d:%d.\n" % (hours, mins, secs))
