import logging
import os
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.variant_tools import import_roi_bed, variant_specific_predict
from maxatac.utilities.prediction_tools import write_predictions_to_bigwig, create_prediction_regions
from maxatac.utilities.genome_tools import build_chrom_sizes_dict
import pandas as pd
import pybedtools


def run_variants(args):
    """Predict TF binding using a variant call format file to make sequence specific predictions
    
    Args:
        args ([type]): input_bigwig, sequence, models, chromosomes, variant_start_pos, nucleotide, overhang, output, name
    """
    output_directory = get_dir(args.output)
    output_bw = os.path.join(output_directory, args.name + ".bw")
    output_bg = os.path.join(output_directory, args.name + ".bg")
    output_windows = os.path.join(output_directory, args.name + "_windows.bed")

    # Import regions to make predictions in. Currently designed for non-overlapping LD blocks or whole chromosome approaches
    if args.roi:
        logging.error("Import prediction regions")
        bed_df = pd.read_table(args.roi, header=None, names=["chr", "start", "stop"])

        regions_pool = import_roi_bed(args.roi)

        # Find the chromosomes for which we can make predictions based on the requested chroms
        # and the BED regions provided in the ROI file
        chrom_list = list(set(args.chromosomes).intersection(set(bed_df["chr"].unique())))

    else:
        logging.error("Create prediction regions")
        bed_df = create_prediction_regions(chromosomes=args.chromosomes,
                                           chrom_sizes=args.chrom_sizes,
                                           blacklist=args.blacklist,
                                           step_size=args.step_size
                                           )

        regions_pool = pybedtools.BedTool.from_dataframe(bed_df)

        chrom_list = args.chromosomes

    logging.error(f"Making sequence specific predictions for: {args.input_bigwig} \n" +
                  f"Writing files with name: {args.name} \n" +
                  f"Output bigwig: {output_bw} \n" +
                  f"Output bedgraph: {output_bg} \n" +
                  f"Prediction Windows: {output_windows} \n" +
                  f"Sequence file: {args.sequence} \n" +
                  f"MaxATAC model: {args.model} \n" +
                  f"Chromosome(s): {chrom_list} \n" +
                  f"Variants Bed: {args.variants_bed}"
                  )

    try:
        os.mkdir(output_directory)

    except OSError as error:
        print(error)

    predictions = variant_specific_predict(args.model,
                                           args.input_bigwig,
                                           args.sequence,
                                           regions_pool,
                                           args.variants_bed)

    predictions.to_csv(output_bg, sep="\t", index=False, header=False)

    chr_dict = build_chrom_sizes_dict(chrom_list, args.chrom_sizes)

    write_predictions_to_bigwig(predictions,
                                output_bw,
                                chrom_sizes_dictionary=chr_dict,
                                chromosomes=chrom_list)
