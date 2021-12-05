import logging
import os
from maxatac.utilities.constants import DEFAULT_CHROM_SIZES
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.variant_tools import SequenceSpecificPrediction
from maxatac.utilities.prediction_tools import write_predictions_to_bigwig
from maxatac.utilities.genome_tools import build_chrom_sizes_dict

def run_variants(args):
    """Predict TF binding using a variant call format file to make sequence specific predictions
    
    Args:
        args ([type]): input_bigwig, sequence, models, chromosomes, variant_start_pos, nucleotide, overhang, output, name
    """    
    output_directory = get_dir(args.output)
    output_bw = os.path.join(output_directory, args.name + ".bw")
    output_bg = os.path.join(output_directory, args.name + ".bg")
    output_windows = os.path.join(output_directory, args.name + "_windows.bed")

    logging.error(f"Making sequence specific predictions for: {args.input_bigwig} \n" +
                  f"Writing files with name: {args.name} \n" +
                  f"Output bigwig: {output_bw} \n" +
                  f"Output bedgraph: {output_bg} \n" +
                  f"Output_windows: {output_windows} \n" +
                  f"Sequence file: {args.sequence} \n" +
                  f"MaxATAC model: {args.model} \n" +
                  f"Chromosome: {args.chromosome} \n" +
                  f"Variant position: {args.variant_start_pos} \n" +
                  f"Changing nucleotide to: {args.nucleotide} \n" +
                  f"Overhang: {args.overhang}")

    # Initialize the object
    maxatac_variants = SequenceSpecificPrediction(signal=args.input_bigwig,
                                                    sequence=args.sequence,)
    
    # Make a sequence specific prediction
    variant_prediction = maxatac_variants.predict(model=args.model, 
                                                  chromosome=args.chromosome, 
                                                  variant_start=args.variant_start_pos, 
                                                  target_nucleotide=args.nucleotide, 
                                                  overhang=args.overhang)
    
    # Write the predition regions to a bed file
    maxatac_variants.windows_df.to_csv(output_windows, sep="\t", index=False, header=False)
    
    # Write the predictions to a bedgraph
    variant_prediction.to_csv(output_bg, sep="\t", index=False, header=False)

    # Build chromosome sizes dictionary
    chrom_sizes_dict = build_chrom_sizes_dict(chromosome_list=[args.chromosome], chrom_sizes_filename=DEFAULT_CHROM_SIZES)
    
    # Write predictions to a bigwig file
    write_predictions_to_bigwig(variant_prediction, 
                                output_bw, 
                                chrom_sizes_dictionary=chrom_sizes_dict, 
                                chromosomes=[args.chromosome])