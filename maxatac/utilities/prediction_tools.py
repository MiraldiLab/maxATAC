import logging
import numpy as np
import pandas as pd
import pybedtools

from maxatac.utilities.system_tools import Mute

with Mute():
    from keras.models import load_model
    from maxatac.utilities.genome_tools import load_bigwig, load_2bit, dump_bigwig, get_one_hot_encoded


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

        input_batch = get_region_values(signal=signal,
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
                                chrom_sizes_dictionary,
                                chromosomes,
                                number_intervals=32
                                ):
    """
    Write the predictions dataframe into a bigwig file

    :param df: The dataframe of BED regions with prediction scores
    :param output_filename: The output bigwig filename
    :param chrom_sizes_dictionary: A dictionary of chromosome sizes used to form the bigwig file
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
        header = [(chromosomes[0], int(chrom_sizes_dictionary[chromosomes[0]]))]
        data_stream.addHeader(header)

        data_stream.addEntries(
            chroms=windowed_coordinates_dataframe["chr"].tolist(),
            starts=windowed_coordinates_dataframe["start"].tolist(),
            ends=windowed_coordinates_dataframe["stop"].tolist(),
            values=windowed_coordinates_dataframe["score"].tolist()
        )


def get_region_values(
        signal,
        sequence,
        average,
        roi_pool,
        input_channels,
        input_length
):
    """
    Get the bigwig values for each ROI in the ROI pool

    :param signal: ATAC-seq signal file
    :param sequence: 2bit DNA file
    :param average: Average ATAC-seq signal file
    :param roi_pool: Pool of regions to predict on
    :param input_channels: Number of input channels
    :param input_length: Length of the input regions

    :return: Array of examples
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


def import_prediction_regions(bed_file, region_length, chromosomes, chrom_sizes_dictionary, blacklist):
    """
    Import a BED file and format the regions to be compatible with our maxATAC models

    :param bed_file: Input BED file to format
    :param region_length: Length of the regions to resize BED intervals to
    :param chromosomes: Chromosomes to filter the BED file for
    :param chrom_sizes_dictionary: A dictionary of chromosome sizes to make sure intervals fall in bounds
    :param blacklist: A BED file of regions to exclude from our analysis

    :return: A dataframe of BED regions compatible with our model
    """
    df = pd.read_csv(bed_file,
                     sep="\t",
                     usecols=[0, 1, 2],
                     header=None,
                     names=["chr", "start", "stop"],
                     low_memory=False)

    df = df[df["chr"].isin(chromosomes)]

    df["length"] = df["stop"] - df["start"]

    df["center"] = np.floor(df["start"] + (df["length"] / 2)).apply(int)

    df["start"] = np.floor(df["center"] - (region_length / 2)).apply(int)

    df["stop"] = np.floor(df["center"] + (region_length / 2)).apply(int)

    df["END"] = df["chr"].map(chrom_sizes_dictionary)

    df = df[df["stop"].apply(int) < df["END"].apply(int)]

    df = df[df["start"].apply(int) > 0]

    df = df[["chr", "start", "stop"]]

    BED_df_bedtool = pybedtools.BedTool.from_dataframe(df)

    blacklist_bedtool = pybedtools.BedTool(blacklist)

    blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

    df = blacklisted_df.to_dataframe()

    df.columns = ["chr", "start", "stop"]

    return df


def get_input_matrix(rows,
                     cols,
                     signal_stream,
                     average_stream,
                     sequence_stream,
                     bp_order,
                     chromosome,
                     start,  # end - start = cols
                     end
                     ):
    """
    Generate the matrix of values from the signal, sequence, and average data tracks

    :param rows: (int) The number of channels or rows
    :param cols: (int) The number of columns or length
    :param signal_stream: (str) ATAC-seq signal
    :param average_stream: (str) Average ATAC-seq signal
    :param sequence_stream: (str) One-hot encoded sequence
    :param bp_order: (list) Order of the bases in matrix
    :param chromosome: (str) Chromosome name
    :param start: (str) Chromosome start
    :param end: (str) Chromosome end

    :return: A matrix that is rows x columns with the values from each file
    """
    input_matrix = np.zeros((rows, cols))

    for n, bp in enumerate(bp_order):
        input_matrix[n, :] = get_one_hot_encoded(
            sequence_stream.sequence(chromosome, start, end),
            bp
        )

    signal_array = np.array(signal_stream.values(chromosome, start, end))
    avg_array = np.array(average_stream.values(chromosome, start, end))

    input_matrix[4, :] = signal_array
    input_matrix[5, :] = input_matrix[4, :] - avg_array

    return input_matrix.T

