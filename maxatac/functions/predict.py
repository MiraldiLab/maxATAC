import logging
import os

from maxatac.utilities.genome_tools import (build_chrom_sizes_dict,
                                            import_bed)

from maxatac.utilities.constants import INPUT_LENGTH, INPUT_CHANNELS

from maxatac.utilities.prediction_tools import make_predictions, write_predictions_to_bigwig

from maxatac.utilities.session import configure_session
from maxatac.utilities.system_tools import get_dir


def run_prediction(args):
    """
    Run prediction on regions of interest using a maxATAC model

    @param args: The arguments list from the argsparser
    @return:
    """
    output_directory = get_dir(args.output_directory)

    outfile_name_bigwig = os.path.join(output_directory, args.prefix + ".bw")

    logging.error("Prediction Parameters \n" +
                  "Output filename: " + outfile_name_bigwig + "\n"
                  "Target signal: " + args.signal + "\n"
                  "Sequence data: " + args.sequence + "\n"
                  "Models: \n   - " + "\n   - ".join(args.models) + "\n"
                  "Chromosomes: " + str(args.predict_chromosomes) + "\n"
                  "Minimum prediction value to be reported: " + str(args.minimum) + "\n"
                  "Threads count: " + str(args.threads) + "\n"
                  "Output directory: " + str(output_directory) + "\n"
                  "Output filename: " + outfile_name_bigwig + "\n"
                  )

    roi_pool = import_bed(bed_file=args.roi,
                          region_length=INPUT_LENGTH,
                          chromosomes=args.predict_chromosomes,
                          chromosome_sizes_dictionary=build_chrom_sizes_dict(args.predict_chromosomes,
                                                                             args.chromosome_sizes
                                                                             ),
                          blacklist=args.blacklist
                          )

    logging.error("Make predictions on ROIs of interest")

    configure_session(args.threads)

    # TODO Write the code so it can make prediction on multiple chromosomes and write them correctly to bigwig files.
    prediction_results = make_predictions(args.signal,
                                          args.sequence,
                                          args.average,
                                          args.models[0],
                                          roi_pool,
                                          args.batch_size,
                                          args.round,
                                          INPUT_CHANNELS,
                                          INPUT_LENGTH
                                          )

    logging.error("Write predictions to a bigwig file")

    write_predictions_to_bigwig(prediction_results,
                                output_filename=outfile_name_bigwig,
                                chromosome_length_dictionary=build_chrom_sizes_dict(args.predict_chromosomes,
                                                                                    args.chromosome_sizes
                                                                                    ),
                                chromosomes=args.predict_chromosomes
                                )
