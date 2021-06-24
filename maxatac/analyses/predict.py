import logging
import os
import pandas as pd
from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict
    from maxatac.utilities.constants import INPUT_CHANNELS, INPUT_LENGTH
    from maxatac.utilities.prediction_tools import write_predictions_to_bigwig, \
        import_prediction_regions, create_prediction_regions, PredictionDataGenerator, make_stranded_predictions
    from maxatac.utilities.session import configure_session


def run_prediction(args):
    """
    Run prediction using a maxATAC model.

    BED file requirements for prediction. You must have at least a 3 column file with chromosome, start,
    and stop coordinates. The interval distance has to be the same as the distance used to train the model. If you
    trained a model with a resolution 1024, you need to make sure your intervals are spaced 1024 bp apart for
    prediction with your model.

    Example input BED file for prediction:

    chr1 | 1000  | 2024
    ___________

    Workflow overview

    1) Create directories and set up filenames
    2) Prepare regions for prediction
    3) Convert predictions to bigwig format and write results

    :param args : output_directory, prefix, signal, sequence, models, predict_chromosomes, minimum, threads, batch_size
    roi, chromosome_sizes, blacklist, average

    :return : A bigwig file of TF binding predictions
    """
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
    if args.roi:
        logging.error("Import prediction regions")

        regions_pool = import_prediction_regions(bed_file=args.roi,
                                                 region_length=INPUT_LENGTH,
                                                 chromosomes=args.chromosomes,
                                                 chrom_sizes_dictionary=build_chrom_sizes_dict(args.chromosomes,
                                                                                               args.chromosome_sizes
                                                                                               ),
                                                 blacklist=args.blacklist
                                                 )

    else:
        logging.error("Create prediction regions")

        regions_pool = create_prediction_regions(region_length=INPUT_LENGTH,
                                                 chromosomes=args.chromosomes,
                                                 chrom_sizes=args.chromosome_sizes,
                                                 blacklist=args.blacklist,
                                                 step_size=args.step_size
                                                 )

    configure_session(1)

    logging.error("Make prediction on forward strand")

    forward_strand_predictions = make_stranded_predictions(signal=args.signal,
                                                           sequence=args.sequence,
                                                           models=args.models[0],
                                                           predict_roi_df=regions_pool,
                                                           batch_size=args.batch_size,
                                                           use_complement=False)

    if args.stranded:
        logging.error("Make prediction on reverse strand")

        reverse_strand_prediction = make_stranded_predictions(signal=args.signal,
                                                              sequence=args.sequence,
                                                              models=args.models[0],
                                                              predict_roi_df=regions_pool,
                                                              batch_size=args.batch_size,
                                                              use_complement=True)

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

        # Write the predictions to a bigwig file
        write_predictions_to_bigwig(merged_predictions,
                                    output_filename=os.path.join(output_directory, args.prefix + "_max.bw"),
                                    chrom_sizes_dictionary=build_chrom_sizes_dict(args.chromosomes,
                                                                                  args.chromosome_sizes
                                                                                  ),
                                    chromosomes=args.chromosomes,
                                    agg_mean=False
                                    )

        # Write the predictions to a bigwig file
        write_predictions_to_bigwig(merged_predictions,
                                    output_filename=os.path.join(output_directory, args.prefix + "_mean.bw"),
                                    chrom_sizes_dictionary=build_chrom_sizes_dict(args.chromosomes,
                                                                                  args.chromosome_sizes
                                                                                  ),
                                    chromosomes=args.chromosomes,
                                    agg_mean=True
                                    )

    else:
        logging.error("Write predictions to a bigwig file")

        # Write the predictions to a bigwig file
        write_predictions_to_bigwig(forward_strand_predictions,
                                    output_filename=os.path.join(output_directory, args.prefix + ".bw"),
                                    chrom_sizes_dictionary=build_chrom_sizes_dict(args.chromosomes,
                                                                                  args.chromosome_sizes
                                                                                  ),
                                    chromosomes=args.chromosomes
                                    )
