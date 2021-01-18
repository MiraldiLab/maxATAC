import logging

from maxatac.utilities.session import configure_session

from maxatac.utilities.constants import (BP_RESOLUTION,
                                         CHR_POOL_SIZE,
                                         INPUT_LENGTH,
                                         INPUT_CHANNELS,
                                         TRAIN_MONITOR,
                                         TRAIN_SCALE_SIGNAL)

from maxatac.utilities.training_tools import DataGenerator

from maxatac.utilities.model_tools import MaxATACModel


def run_training(args):
    """
    Train a maxATAC model

    :param args: The argument parser object with the parameters from the parser
    :return: A trained maxATAC model
    """
    # We configure the session to have the generator handle the multi-processing
    configure_session(1)

    logging.error("Loading model with parameters: \n"
                  "Seed: " + str(args.seed) + "\n" +
                  "Average: " + args.average + "\n" +
                  "Output Directory: " + args.output + "\n" +
                  "Filename Prefix: " + args.prefix + "\n" +
                  "Number of Filters: " + str(args.number_of_filters) + "\n" +
                  "Kernel Size in BP: " + str(args.kernel_size) + "\n" +
                  "Scale filters by this factor each layer: " + str(args.filter_scaling_factor) + "\n" +
                  "Number of threads to use: " + str(args.threads) + "\n")

    # Initialize the model
    maxatac_model = MaxATACModel(arch="DCNN_V2",
                                 seed=args.seed,
                                 output_directory=args.output,
                                 prefix=args.prefix,
                                 number_of_filters=args.number_of_filters,
                                 kernel_size=args.kernel_size,
                                 filter_scaling_factor=args.filter_scaling_factor,
                                 threads=args.threads,
                                 training_monitor=TRAIN_MONITOR,
                                 meta_path=args.meta_file)

    logging.error("Initializing the training generator with the parameters: \n" +
                  "Training random ratio proportion: " + str(args.train_rand_ratio) + "\n" +
                  "Training chromosomes: " + str(args.train_chroms) + "\n" +
                  "Training batch size: " + str(args.train_batch_size) + "\n")

    # Initialize the training generator
    train_data_generator = DataGenerator(sequence=args.sequence,
                                         average=args.average,
                                         meta_dataframe=maxatac_model.meta_dataframe,
                                         random_ratio=args.train_rand_ratio,
                                         chromosomes=args.train_chroms,
                                         batch_size=args.train_batch_size,
                                         blacklist=args.blacklist,
                                         chromosome_sizes=args.chrom_sizes,
                                         chromosome_pool_size=CHR_POOL_SIZE,
                                         bp_resolution=BP_RESOLUTION,
                                         region_length=INPUT_LENGTH,
                                         input_channels=INPUT_CHANNELS,
                                         cell_types=maxatac_model.cell_types,
                                         peak_paths=maxatac_model.peak_paths,
                                         batches_per_epoch=args.train_steps_per_epoch,
                                         scale_signal=None)

    logging.error("Initializing the validation generator with the parameters: \n" +
                  "Validation random ratio proportion: " + str(args.validate_rand_ratio) + "\n" +
                  "Validation chromosomes: " + str(args.validate_chroms) + "\n" +
                  "Validation batch size: " + str(args.train_batch_size) + "\n")

    # Initialize the validation data generator
    validate_data_generator = DataGenerator(sequence=args.sequence,
                                            average=args.average,
                                            meta_dataframe=maxatac_model.meta_dataframe,
                                            random_ratio=args.validate_rand_ratio,
                                            chromosomes=args.validate_chroms,
                                            batch_size=args.validate_batch_size,
                                            blacklist=args.blacklist,
                                            chromosome_sizes=args.chrom_sizes,
                                            chromosome_pool_size=CHR_POOL_SIZE,
                                            bp_resolution=BP_RESOLUTION,
                                            region_length=INPUT_LENGTH,
                                            input_channels=INPUT_CHANNELS,
                                            cell_types=maxatac_model.cell_types,
                                            peak_paths=maxatac_model.peak_paths,
                                            batches_per_epoch=args.validate_steps_per_epoch,
                                            scale_signal=None)

    logging.error("Fitting the maxATAC model with parameters: \n"
                  "Epochs: " + str(args.epochs) + "\n"
                  "Training batches: " + str(args.train_steps_per_epoch) + "\n"
                  "Validation batches: " + str(args.validate_steps_per_epoch) + "\n")

    # Fit the model 
    maxatac_model.fit_model(train_gen=train_data_generator,
                            val_gen=validate_data_generator,
                            epochs=args.epochs)

    # Plot model structure and metrics is plot option is True
    if args.plot:
        logging.error("Plotting Results")

        maxatac_model.export_model_structure()

        maxatac_model.plot_metrics('dice_coef')
        maxatac_model.plot_metrics("acc")
        maxatac_model.plot_metrics("loss")

    logging.error("Results are saved to: " + maxatac_model.results_location)
