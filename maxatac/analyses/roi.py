import logging

from maxatac.utilities.genome_tools import build_chrom_sizes_dict
from maxatac.utilities.roi_tools import ROIPool, ValidationData


def run_roi(args):
    logging.error("Generating training regions of interest file: \n" +
                  "Meta Path: " + args.meta_file + "\n" +
                  "Output Directory: " + args.output_dir + "\n" +
                  "Training chromosomes: \n   - " + "\n   - ".join(args.train_chroms) + "\n")

    # The ROI pool is used to import the training data based on the meta file and window size
    training_data = ROIPool(meta_path=args.meta_file,
                            region_length=1024,
                            chromosomes=args.train_chroms,
                            chromosome_sizes_dictionary=build_chrom_sizes_dict(args.train_chroms,
                                                                               args.chromosome_sizes),
                            blacklist=args.blacklist)

    logging.error("Writing training ROI files to BED")

    # Write the ROI pool stats and BED files
    training_data.write_ROI_pools(prefix=args.training_prefix,
                                  output_dir=args.output_dir)

    logging.error("Generating validation regions of interest file: \n" +
                  "Meta Path: " + args.meta_file + "\n" +
                  "Validation random ratio proportion: " + str(args.validate_random_ratio) + "\n" +
                  "Validation chromosomes: \n   - " + "\n   - ".join(args.validate_chroms) + "\n"
                                                                                             "Blacklist Regions BED: " + args.blacklist + "\n" +
                  "Preferences: " + args.preferences)

    # Import the validation data using the meta file
    validation_data = ValidationData(meta_path=args.meta_file,
                                     region_length=1024,
                                     chromosomes=args.validate_chroms,
                                     chromosome_sizes=args.chromosome_sizes,
                                     blacklist=args.blacklist,
                                     random_ratio=args.validate_random_ratio,
                                     preferences=args.preferences)

    logging.error("Writing validation ROI files to BED")

    # Write the validation pool BED, TSV, and stats files
    validation_data.write_validation_pool(prefix=args.validation_prefix,
                                          output_dir=args.output_dir)

    logging.error("Done")
