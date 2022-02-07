import numpy as np
from maxatac.utilities.constants import BP_ORDER
from maxatac.utilities.genome_tools import load_bigwig, load_2bit, get_one_hot_encoded
from tensorflow.keras.models import load_model
from Bio.Seq import Seq
import pandas as pd
import pybedtools


def get_roi_variant_overlap(variants_bed: str, roi_BT: pybedtools.BedTool):
    """Generate a summary of variant overlaps with ROIs

    Args:
        variants_bed (str): Path to variant bed file
        roi_BT (str): ROI bedtool object

    Returns:
        pd.Dataframe: A dataframe of roi and variant intersections annotated with the nucleotide and index to be changed.
        
    Example:
    >>> intersection_df = get_roi_variant_overlap(variants_bed, roi_BT)
    """
    # Import variants bed as bedtools object
    variants_BT = pybedtools.BedTool(variants_bed)

    # Intersect ROI with variants bed file and convert to dataframe
    intersect_df = roi_BT.intersect(variants_BT, loj=True).to_dataframe(names=["chr", "start", "stop", "rs_chr", "rs_start", "rs_stop", "nucleotide"])
    
    row_indices = []

    for _ , row in intersect_df.iterrows():
        if row["rs_start"] == -1:
            row_indices.append(-1)
        else:
            row_indices.append(row["rs_start"] - row["start"])
            
    # Find the index of the array to be changed during prediction
    intersect_df["index"] = row_indices
    
    return intersect_df


def get_seq_specific_input_matrix(window, 
                                  signal: str, 
                                  sequence: str,
                                  BP_DICT={"A":0, "C":1, "G":2, "T":3},
                                  bp_order=BP_ORDER,
                                  cols=1024,
                                  rows=5):
    """Get a sequence specific input matrix

    Args:
        window (pd.Series): The window annotated with the index and nucleotide to switch
        signal (str): ATAC-seq signal path
        sequence (str): Path to 2bit DNA file
        BP_DICT (dict, optional): Dictionary of nucleotide and row position. Defaults to {"A":0, "C":1, "G":2, "T":3}.
        bp_order (list, optional): Order of nucleotides. Defaults to BP_ORDER.
        cols (int, optional): Region length. Defaults to 1024.
        rows (int, optional): Number of channels. Defaults to 6.

    Returns:
        np.Array: A numpy array that corresponds to the ATAC-seq and sequence of the region of interest
    """
    # The nucleotide positions are 0-4
    signal_stream = load_bigwig(signal)
    sequence_stream  = load_2bit(sequence)
    
    input_matrix = np.zeros((rows, cols))
    
    for n, bp in enumerate(bp_order):
        # Get the sequence from the interval of interest
        target_sequence = Seq(sequence_stream.sequence(window['chr'], window['start'], window['stop']))

        # Get the one hot encoded sequence
        input_matrix[n, :] = get_one_hot_encoded(target_sequence, bp)

    signal_array = np.array(signal_stream.values(window['chr'], window['start'], window['stop']))

    input_matrix[4, :] = signal_array

    input_matrix = input_matrix.T
            
    if window["index"] == -1:  
        pass
    
    else:
        # Get the nucleotide row
        nucleotide_index = BP_DICT[window["nucleotide"]]
    
        input_matrix[window["index"], nucleotide_index] = 1

        other_nucleotides = [item for item in [0,1,2,3] if item not in [nucleotide_index]]
        
        input_matrix[window["index"], other_nucleotides[0]] = 0
        input_matrix[window["index"], other_nucleotides[1]] = 0
        input_matrix[window["index"], other_nucleotides[2]] = 0

    return np.array([input_matrix])


def convert_predictions_to_bedgraph(predictions: list, 
                                    predict_roi_df: pd.DataFrame):
    """Convert output predictions to bedgraph

    Args:
        predictions (list): list of prediction arrays
        predict_roi_df (pd.DataFrame): dataframe with the ROI information

    Returns:
        pd.DataFrame: Dataframe of predictions in bedgraph format
    """
    predictions_df = pd.DataFrame(data=predictions, index=None)
    
    predictions_df["chr"] = predict_roi_df["chr"]
    predictions_df["start"] = predict_roi_df["start"]
    predictions_df["stop"] = predict_roi_df["stop"]

    # Create BedTool object from the dataframe
    coordinates_dataframe = pybedtools.BedTool.from_dataframe(predictions_df[['chr', 'start', 'stop']])

    # Window the intervals into 32 bins
    windowed_coordinates = coordinates_dataframe.window_maker(b=coordinates_dataframe, n=32)

    # Create a dataframe from the BedTool object
    windowed_coordinates_dataframe = windowed_coordinates.to_dataframe()

    # Drop all columns except those that have scores in them
    scores_dataframe = predictions_df.drop(['chr', 'start', 'stop'], axis=1)

    # Take the scores and reshape them into a column of the pandas dataframe
    windowed_coordinates_dataframe['score'] = scores_dataframe.to_numpy().flatten()

    # Rename the columns of the dataframe
    windowed_coordinates_dataframe.columns = ['chr', 'start', 'stop', 'score']

    # Get the mean of all sliding window predicitons
    windowed_coordinates_dataframe = windowed_coordinates_dataframe.groupby(["chr", "start", "stop"], as_index=False).mean()

    return windowed_coordinates_dataframe


def import_roi_bed(roi_bed):
    """Import bed file of haplotype blocks or LD blocks

    Args:
        roi_bed (str): Path to the bedfile with region information

    Returns:
        pyBedTool.BedTool: A BedTool object of regions to use for prediction
    """
    roi_bedtool = pybedtools.BedTool(roi_bed)
    roi_bedtool_slop = roi_bedtool.slop(g=chrom_sizes,b=512)

    roi_DF = pybedtools.BedTool().window_maker(b=roi_bedtool_slop, w=1024, s=256).to_dataframe()
    roi_DF["length"] = roi_DF["end"] - roi_DF["start"]
    roi_DF = roi_DF[roi_DF["length"] == 1024]
    
    roi_BT = pybedtools.BedTool.from_dataframe(roi_DF[["chrom", "start", "end"]])

    return roi_BT


def variant_specific_predict(model:str,
            signal:str,
            sequence:str,
            roi_BT:pybedtools.BedTool,
            variants_bed:str):
    """Make predictions in LD blocks

    Args:
        model (str): [description]
        signal (str): [description]
        sequence (str): [description]
        roi_BT (str): [description]
        variants_bed (str): [description]

    Returns:
        [type]: [description]
    """
    # Load the maxATAC model
    nn_model = load_model(model, compile=False)

    # Get the overlap between variants and prediction windows and annotated with index and nucleotide to change
    prediction_windows = get_roi_variant_overlap(variants_bed, roi_BT)
    
    prediction_list = []
    
    for _, window in prediction_windows.iterrows():
        # Get the sequence specific input matrix for the genomic region of interest
        seq_specific_array = get_seq_specific_input_matrix(window, signal, sequence)
        
        # Append the batch of predictions from the model
        prediction_list.append(nn_model.predict_on_batch(seq_specific_array).flatten())
        
    # convert all predictions to a bedgraph format dataframe
    bedgraph_df = convert_predictions_to_bedgraph(predictions=prediction_list, 
                                                            predict_roi_df = prediction_windows)    
    return bedgraph_df
