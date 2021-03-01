import logging
from maxatac.utilities.system_tools import Mute

with Mute():
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict
    from maxatac.utilities.roi_tools import GenomicRegions


def run_roi(args):
    """
    Generate the training and validation pools of genomic regions

    This method will use the run meta file to merge the ATAC-seq peaks and ChIP-seq peaks into BED files where each
    entry is an example region. ONE MAJOR ASSUMPTION IS THAT THE TEST CELL LINE IS NOT INCLUDED IN THE META FILE!!!!!

    The input meta file must have the columns in any order:

    TF | Cell_Line | ATAC_Signal_File | Binding_File | ATAC_Peaks | ChIP_peaks
    _________________
    Workflow Overview

    1) TrainingData: Import the ATAC-seq and ChIP-seq and filter for training chromosomes
    2) TrainingData.write_data: writes the ChIP, ATAC, and combined ROI pools with stats
    3) ValidationData: Import the ATAC-seq and ChIP-seq and filter for validation chromosomes
    4) ValidationData.write_data: writes the ChIP, ATAC, and combined ROI pools with stats

    :param args: meta_file, output_dir, train_chroms, blacklist, training_prefix, validate_random_ratio,
    validate_chroms, chromosome_sizes, preferences, threads, validation_prefix

    :return: BED files for input into training with associated statistics
    """
    logging.error("Generating training regions of interest file: \n" +
                  "Meta Path: " + args.meta_file + "\n" +
                  "Output Directory: " + args.output_dir + "\n" +
                  "Training chromosomes: \n   - " + "\n   - ".join(args.train_chroms) + "\n")

    # The ROI pool is used to import the training data based on the meta file and window size
    training_data = GenomicRegions(meta_path=args.meta_file,
                                   region_length=1024,
                                   chromosomes=args.train_chroms,
                                   chromosome_sizes_dictionary=build_chrom_sizes_dict(args.train_chroms,
                                                                                    args.chromosome_sizes),
                                   blacklist=args.blacklist)

    logging.error("Writing training ROI files to BED")

    # Write the ROI pool stats and BED files
    training_data.write_data(prefix=args.training_prefix,
                             output_dir=args.output_dir,
                             set_tag="training")

    logging.error("Generating validation regions of interest file: \n" +
                  "Meta Path: " + args.meta_file + "\n" +
                  "Validation chromosomes: \n   - " + "\n   - ".join(args.validate_chroms) + "\n" +
                  "Blacklist Regions BED: " + args.blacklist + "\n")

    # Import the validation data using the meta file
    validation_data = GenomicRegions(meta_path=args.meta_file,
                                     region_length=1024,
                                     chromosomes=args.validate_chroms,
                                     chromosome_sizes_dictionary=build_chrom_sizes_dict(args.validate_chroms,
                                                                                      args.chromosome_sizes),
                                     blacklist=args.blacklist)

    logging.error("Writing validation ROI files to BED")

    # Write the validation pool BED, TSV, and stats files
    validation_data.write_data(prefix=args.validation_prefix,
                               output_dir=args.output_dir,
                               set_tag="validation")

    logging.error("Done")
