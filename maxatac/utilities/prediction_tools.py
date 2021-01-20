from keras.models import load_model
import logging
import numpy as np
import pandas as pd
import pybedtools

from maxatac.utilities.genome_tools import load_bigwig, load_2bit, get_input_matrix, dump_bigwig, import_bed


def make_predictions(
        signal,
        sequence,
        average,
        model,
        predict_roi_df,
        batch_size,
        round_predictions,
        input_channels,
        input_length,
        predictions=""
):
    """
    Make predictions on the input ROIs

    :param signal: The ATAC-seq signal file
    :param sequence: The 2bit DNA sequence file
    :param average: The average ATAC-seq signal file
    :param model: The trained maxATAC model(s)
    :param predict_roi_df: A dataframe containing a BED formatted dataframe
    :param batch_size: The number of examples to predict in each batch
    :param round_predictions: Number of decimal places to round prediction scores to
    :param input_channels: Number of input channels
    :param input_length: Length of the input regions of interest
    :param predictions: empty and used to hold predictions.

    :return: A dataframe of scores associated with the regions of interest
    """
    logging.error("Load Model")

    # Load the neural network model that has been trained.
    nn_model = load_model(model, compile=False)

    # Determine the number of batches based on the total number of regions and batch size
    n_batches = int(predict_roi_df.shape[0] / batch_size)

    # Get batches of indexes
    all_batch_idxs = np.array_split(np.arange(predict_roi_df.shape[0]), n_batches)

    logging.error("Making predictions")

    # Loop through all of the batches and make predictions.
    # TODO Make this loop into a parallel loop
    for idx, batch_idxs in enumerate(all_batch_idxs):
        batch_roi_df = predict_roi_df.loc[batch_idxs, :]

        batch_roi_df.reset_index(drop=True, inplace=True)

        input_batch = get_roi_values(signal=signal,
                                     sequence=sequence,
                                     average=average,
                                     input_channels=input_channels,
                                     input_length=input_length,
                                     roi_pool=batch_roi_df
                                     )

        pred_output_batch = nn_model.predict(input_batch)

        if idx == 0:
            predictions = pred_output_batch

        else:
            predictions = np.vstack((predictions, pred_output_batch))

    predictions = np.round(predictions, round_predictions)

    logging.error("Parsing results into pandas dataframe")

    predictions_df = pd.DataFrame(data=predictions, index=None, columns=None)

    predictions_df["chr"] = predict_roi_df["chr"]
    predictions_df["start"] = predict_roi_df["start"]
    predictions_df["stop"] = predict_roi_df["stop"]

    return predictions_df


def write_predictions_to_bigwig(df,
                                output_filename,
                                chromosome_length_dictionary,
                                chromosomes,
                                number_intervals=32
                                ):
    """
    Write the predictions dataframe into a bigwig file

    :param df: The dataframe of BED regions with prediction scores
    :param output_filename: The output bigwig filename
    :param chromosome_length_dictionary: A dictionary of chromosome sizes used to form the bigwig file
    :param chromosomes: A list of chromosomes that you are predicting in
    :param number_intervals: The number of 32 bp intervals found in the sequence

    :return: Writes a bigwig file
    """
    # Create BedTool object from the dataframe
    coordinates_dataframe = pybedtools.BedTool.from_dataframe(df[['chr', 'start', 'stop']])

    # Window the intervals into 32 bins
    windowed_coordinates = coordinates_dataframe.window_maker(b=coordinates_dataframe, n=number_intervals)

    # Create a dataframe from the BedTool object
    windowed_coordinates_dataframe = windowed_coordinates.to_dataframe()

    # Drop all columns except those that have scores in them
    scores_dataframe = df.drop(['chr', 'start', 'stop'], axis=1)

    # Take the scores and reshape them into a column of the pandas dataframe
    windowed_coordinates_dataframe['score'] = scores_dataframe.to_numpy().flatten()

    # Rename the columns of the dataframe
    windowed_coordinates_dataframe.columns = ['chr', 'start', 'stop', 'score']

    # TODO parallel write many chromosomes. Hardcoded now to first chromosome in list
    with dump_bigwig(output_filename) as data_stream:
        header = [(chromosomes[0], int(chromosome_length_dictionary[chromosomes[0]]))]
        data_stream.addHeader(header)

        data_stream.addEntries(
            chroms=windowed_coordinates_dataframe["chr"].tolist(),
            starts=windowed_coordinates_dataframe["start"].tolist(),
            ends=windowed_coordinates_dataframe["stop"].tolist(),
            values=windowed_coordinates_dataframe["score"].tolist()
        )


def get_roi_values(
        signal,
        sequence,
        average,
        roi_pool,
        input_channels,
        input_length
):
    """
    Get the bigwig values for each ROI in the ROI pool

    @param signal: ATAC-seq signal file
    @param sequence: 2bit DNA file
    @param average: Average ATAC-seq signal file
    @param roi_pool: Pool of regions to predict on
    @param input_channels: Number of input channels
    @param input_length: Length of the input regions

    @return: Array of examples
    """
    inputs_batch, targets_batch = [], []

    roi_size = roi_pool.shape[0]

    with load_bigwig(signal) as signal_stream, \
            load_2bit(sequence) as sequence_stream, \
            load_bigwig(average) as average_stream:
        for row_idx in range(roi_size):
            row = roi_pool.loc[row_idx, :]

            chromosome = row[0]

            start = int(row[1])

            end = int(row[2])

            input_matrix = get_input_matrix(rows=input_channels,
                                            cols=input_length,
                                            bp_order=["A", "C", "G", "T"],
                                            average_stream=average_stream,
                                            signal_stream=signal_stream,
                                            sequence_stream=sequence_stream,
                                            chromosome=chromosome,
                                            start=start,
                                            end=end
                                            )

            inputs_batch.append(input_matrix)

    return np.array(inputs_batch)