import logging
import os
import timeit
import pandas as pd
import multiprocessing
from multiprocessing import Pool, Manager
from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict
    from maxatac.utilities.constants import INPUT_LENGTH
    from maxatac.utilities.prediction_tools import write_predictions_to_bigwig, \
        import_prediction_regions, create_prediction_regions, make_stranded_predictions

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
    3) Make predictions on the reference strand. 
    3) Convert predictions to bigwig format and write results.

    Args:
        output_directory, prefix, signal, sequence, models, predict_chromosomes, threads, batch_size, roi, chromosome_sizes, blacklist, average

    Returns:
        A bigwig file of TF binding predictions
    """
    # Start Timer
    startTime = timeit.default_timer()

    # Create the output directory set by the parser
    output_directory = get_dir(args.output)

    # Output filename for the bigwig predictions file based on the output directory and the prefix. Add the bw extension
    outfile_name_bigwig = os.path.join(output_directory, args.prefix + ".bw")

    # The function build_chrom_sizes_dict is used to make sure regions fall within chromosome bounds.
    # Create a dictionary of chromosome sizes used to make the bigwig files
    chrom_sizes_dict = build_chrom_sizes_dict(args.chromosomes, args.chromosome_sizes)
    
    # Import the regions for prediction.
    if args.roi:
        logging.error("Import prediction regions")
        regions_pool = import_prediction_regions(bed_file=args.roi,
                                                 chromosomes=args.chromosomes,
                                                 chrom_sizes_dictionary=chrom_sizes_dict,
                                                 blacklist=args.blacklist
                                                 )
        
        # Find the chromosomes for which we can make predictions based on the requested chroms
        # and the BED regions provided in the ROI file
        chrom_list = list(set(args.chromosomes).intersection(set(regions_pool["chr"].unique())))
        
    else:
        logging.error("Create prediction regions")
        regions_pool = create_prediction_regions(chromosomes=args.chromosomes,
                                                 chrom_sizes=args.chromosome_sizes,
                                                 blacklist=args.blacklist,
                                                 step_size=args.step_size
                                                 )
        
        chrom_list = args.chromosomes
    
    logging.error("Prediction Parameters \n" +
                "Output filename: " + outfile_name_bigwig + "\n" +
                "Target signal: " + args.signal + "\n" +
                "Sequence data: " + args.sequence + "\n" +
                "Model: args.model" + "\n" +
                "Chromosome requested: \n   - " + "\n    -".join(args.chromosomes) + "\n" +
                "Chromosomes in final prediction set: \n   - " + "\n    -".join(chrom_list) + "\n" +
                "Threads count: " + str(args.threads) + "\n" +
                "Output directory: " + str(output_directory) + "\n" +
                "Batch Size: " + str(args.batch_size) + "\n" +
                "Output filename: " + outfile_name_bigwig + "\n"
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

    logging.error("Write predictions to a bigwig file")

    # Write the predictions to a bigwig file and add name to args
    prediction_bedgraph = pd.concat(forward_strand_predictions)
    
    write_predictions_to_bigwig(prediction_bedgraph,
                                output_filename=outfile_name_bigwig,
                                chrom_sizes_dictionary=chrom_sizes_dict,
                                chromosomes=chrom_list
                                )
    
    # If a cutoff file is provided, call peaks
    if args.cutoff_file:
        args.input_bigwig = outfile_name_bigwig

        # Call peaks using specified cutoffs
        run_call_peaks(args)

    # Measure end time of training
    stopTime = timeit.default_timer()
    totalTime = stopTime - startTime

    # Output running time in a nice format.
    mins, secs = divmod(totalTime, 60)
    hours, mins = divmod(mins, 60)

    logging.error("Total Prediction time: %d:%d:%d.\n" % (hours, mins, secs))
