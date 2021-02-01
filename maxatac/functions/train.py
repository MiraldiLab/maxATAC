import logging

from maxatac.architectures.dcnn import get_callbacks
from maxatac.utilities.session import configure_session

from maxatac.utilities.constants import (BP_RESOLUTION,
                                         INPUT_LENGTH,
                                         INPUT_CHANNELS,
                                         TRAIN_MONITOR,
                                         TRAIN_SCALE_SIGNAL)

from maxatac.utilities.training_tools import TrainingDataGenerator, MaxATACModel, ValidationDataGenerator, plot_metrics


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
                  "Output Directory: " + args.output + "\n" +
                  "Filename Prefix: " + args.prefix + "\n" +
                  "Number of Filters: " + str(args.number_of_filters) + "\n" +
                  "Kernel Size in BP: " + str(args.kernel_size) + "\n" +
                  "Scale filters by this factor each layer: " + str(args.filter_scaling_factor) + "\n" +
                  "Number of threads to use: " + str(args.threads) + "\n" +
                  "Initializing the training generator with the parameters: \n" +
                  "Training random ratio proportion: " + str(args.train_rand_ratio) + "\n" +
                  "Training chromosomes: " + str(args.train_chroms) + "\n" +
                  "Training batch size: " + str(args.train_batch_size) + "\n" +
                  "Fitting the maxATAC model with parameters: \n"
                  "Epochs: " + str(args.epochs) + "\n" +
                  "Training batches: " + str(args.train_steps_per_epoch) + "\n")

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

    # Initialize the training generator
    train_data_generator = TrainingDataGenerator(sequence=args.sequence,
                                                 meta_dataframe=maxatac_model.meta_dataframe,
                                                 random_ratio=args.train_rand_ratio,
                                                 chromosomes=args.train_chroms,
                                                 batch_size=args.train_batch_size,
                                                 chromosome_sizes=args.chrom_sizes,
                                                 bp_resolution=BP_RESOLUTION,
                                                 region_length=INPUT_LENGTH,
                                                 input_channels=INPUT_CHANNELS,
                                                 cell_types=maxatac_model.cell_types,
                                                 scale_signal=TRAIN_SCALE_SIGNAL,
                                                 preferences=args.preferences,
                                                 roi_dataframe=args.train_roi)

    # Initialize the validation data generator
    validate_data_generator = ValidationDataGenerator(sequence=args.sequence,
                                                      meta_dataframe=maxatac_model.meta_dataframe,
                                                      bp_resolution=BP_RESOLUTION,
                                                      region_length=INPUT_LENGTH,
                                                      input_channels=INPUT_CHANNELS,
                                                      scale_signal=TRAIN_SCALE_SIGNAL,
                                                      roi_dataframe=args.validate_roi,
                                                      batch_size=args.validate_batch_size)
    # Fit the model
    training_history = maxatac_model.nn_model.fit_generator(generator=train_data_generator.batch_generator(),
                                                            validation_data=validate_data_generator,
                                                            epochs=args.epochs,
                                                            callbacks=get_callbacks(
                                                                model_location=maxatac_model.results_location,
                                                                log_location=maxatac_model.log_location,
                                                                tensor_board_log_dir=maxatac_model.tensor_board_log_dir,
                                                                monitor=maxatac_model.training_monitor
                                                            ),
                                                            steps_per_epoch=args.train_steps_per_epoch,
                                                            use_multiprocessing=maxatac_model.threads > 1,
                                                            workers=maxatac_model.threads,
                                                            verbose=1
                                                            )

    # Plot model structure and metrics if plot option is True
    if args.plot:
        logging.error("Plotting Results")

        maxatac_model.export_model_structure()

        plot_metrics(training_history=training_history,
                     results_location=maxatac_model.results_location,
                     metric='dice_coef')
        plot_metrics(training_history=training_history,
                     results_location=maxatac_model.results_location,
                     metric="binary_accuracy")
        plot_metrics(training_history=training_history,
                     results_location=maxatac_model.results_location,
                     metric="acc")

    logging.error("Results are saved to: " + maxatac_model.results_location)
