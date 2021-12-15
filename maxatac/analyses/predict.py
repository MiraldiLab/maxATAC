import logging
import os
import timeit
import pandas as pd
import multiprocessing
from multiprocessing import Pool, Manager
from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict
    from maxatac.utilities.constants import INPUT_CHANNELS, INPUT_LENGTH
    from maxatac.utilities.prediction_tools import write_predictions_to_bigwig, \
        import_prediction_regions, create_prediction_regions, PredictionDataGenerator, make_stranded_predictions

    from maxatac.analyses.peaks import run_call_peaks


def run_prediction(args):
    """
    Run prediction using a maxATAC model. The user can either provide a bed file of regions to predict on or prediction
    regions can be created based on the chromosome of interest.

    BED file requirements for prediction. You must have at least a 3 column file with chromosome, start,
    and stop coordinates. The interval distance has to be the same as the distance used to train the model. If you
    trained a model with a resolution 1024.

    The user can decide whether to make only predictions on the forward strand or also make prediction on the reverse
    strand. If the user wants both strand, signal tracks will be produced for the forward, reverse, and mean-combined
    bigwig signal tracks will be produced.

    Example input BED file for prediction:

    chr1 | 1000  | 2024
    ___________

    Workflow overview

    1) Create directories and set up filenames
    2) Prepare regions for prediction. Either import user defined regions or create regions based on chromosomes list.
    3) Make predictions on the reference strand. If the reverse predictions are specified also make those signal tracks
    3) Convert predictions to bigwig format and write results

    :param args : output_directory, prefix, signal, sequence, models, predict_chromosomes, threads, batch_size
    roi, chromosome_sizes, blacklist, average

    :return : A bigwig file of TF binding predictions
    """
    # Start Timer
    startTime = timeit.default_timer()

    # Create the output directory set by the parser
    output_directory = get_dir(args.output)

    # Output filename for the bigwig predictions file based on the output directory and the prefix. Add the bw extension
    outfile_name_bigwig = os.path.join(output_directory, args.prefix + ".bw")

    logging.error("Prediction Parameters \n" +
                  "Output filename: " + outfile_name_bigwig + "\n" +
                  "Target signal: " + args.signal + "\n" +
                  "Sequence data: " + args.sequence + "\n" +
                  "Models: \n   - " + "\n   - ".join(args.models) + "\n" +
                  "Threads count: " + str(args.threads) + "\n" +
                  "Output directory: " + str(output_directory) + "\n" +
                  "Batch Size: " + str(args.batch_size) + "\n" +
                  "Output filename: " + outfile_name_bigwig + "\n"
                  )

    # Import the regions for prediction.
    # The function build_chrom_sizes_dict is used to make sure regions fall within chromosome bounds.

    logging.error("Make prediction on forward strand")

    with Pool(int(multiprocessing.cpu_count())) as p:
        forward_strand_predictions = p.starmap(make_stranded_predictions,
                                               [(args.roi,
                                                 INPUT_LENGTH,
                                                 args.chromosomes,
                                                 args.chromosome_sizes,
                                                 args.blacklist,
                                                 args.step_size,
                                                 args.signal,
                                                 args.sequence,
                                                 args.models[0],
                                                 args.batch_size,
                                                 False) for chromosome in args.chromosomes])

    forward_strand_predictions = forward_strand_predictions[0]

    if args.stranded:
        logging.error("Make prediction on reverse strand")

        with Pool(int(multiprocessing.cpu_count())) as p:
            reverse_strand_prediction = p.starmap(make_stranded_predictions,
                                                   [(args.roi,
                                                     INPUT_LENGTH,
                                                     args.chromosomes,
                                                     args.chromosome_sizes,
                                                     args.blacklist,
                                                     args.step_size,
                                                     args.signal,
                                                     args.sequence,
                                                     args.models[0],
                                                     args.batch_size,
                                                     True) for chromosome in args.chromosomes])

        reverse_strand_prediction = reverse_strand_prediction[0]

        logging.error("Write predictions to a bigwig file")

        # Write the predictions to a bigwig file
        write_predictions_to_bigwig(forward_strand_predictions,
                                    output_filename=os.path.join(output_directory, args.prefix + "_forward.bw"),
                                    chrom_sizes_dictionary=build_chrom_sizes_dict(args.chromosomes,
                                                                                  args.chromosome_sizes
                                                                                  ),
                                    chromosomes=args.chromosomes
                                    )

        # Write the predictions to a bigwig file
        write_predictions_to_bigwig(reverse_strand_prediction,
                                    output_filename=os.path.join(output_directory, args.prefix + "_reverse.bw"),
                                    chrom_sizes_dictionary=build_chrom_sizes_dict(args.chromosomes,
                                                                                  args.chromosome_sizes
                                                                                  ),
                                    chromosomes=args.chromosomes
                                    )

        # Merge both strand predictions
        merged_predictions = pd.concat([forward_strand_predictions, reverse_strand_prediction])

        outfile_name_bigwig_max = os.path.join(output_directory, args.prefix + "_max.bw")
        args.input_bigwig = outfile_name_bigwig_max

        # Write the predictions to a bigwig file
        write_predictions_to_bigwig(merged_predictions,
                                    output_filename=outfile_name_bigwig_max,
                                    chrom_sizes_dictionary=build_chrom_sizes_dict(args.chromosomes,
                                                                                  args.chromosome_sizes
                                                                                  ),
                                    chromosomes=args.chromosomes,
                                    agg_mean=False
                                    )
        # Call Peaks using specified cutoffs
        run_call_peaks(args)


        outfile_name_bigwig_mean = os.path.join(output_directory, args.prefix + "_mean.bw")
        args.input_bigwig = outfile_name_bigwig_mean

        # Write the predictions to a bigwig file
        write_predictions_to_bigwig(merged_predictions,
                                    output_filename=outfile_name_bigwig_mean,
                                    chrom_sizes_dictionary=build_chrom_sizes_dict(args.chromosomes,
                                                                                  args.chromosome_sizes
                                                                                  ),
                                    chromosomes=args.chromosomes,
                                    agg_mean=True
                                    )
        # Call Peaks using specified cutoffs
        run_call_peaks(args)

    else:
        logging.error("Write predictions to a bigwig file")

        # Write the predictions to a bigwig file and add name to args
        args.input_bigwig = outfile_name_bigwig

        write_predictions_to_bigwig(forward_strand_predictions,
                                    output_filename=outfile_name_bigwig,
                                    chrom_sizes_dictionary=build_chrom_sizes_dict(args.chromosomes,
                                                                                  args.chromosome_sizes
                                                                                  ),
                                    chromosomes=args.chromosomes
                                    )
        # Call Peaks using specified cutoffs
        run_call_peaks(args)

    # Measure End Time of Training
    stopTime = timeit.default_timer()
    totalTime = stopTime - startTime

    # Output running time in a nice format.
    mins, secs = divmod(totalTime, 60)
    hours, mins = divmod(mins, 60)

    logging.error("Total Prediction time: %d:%d:%d.\n" % (hours, mins, secs))
