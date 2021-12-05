import logging
import numpy as np
from maxatac.utilities.constants import INPUT_CHANNELS, INPUT_LENGTH, BP_ORDER
from maxatac.utilities.genome_tools import load_bigwig, load_2bit, get_one_hot_encoded
from tensorflow.keras.models import load_model
from Bio.Seq import Seq
import pandas as pd
import pybedtools

class SequenceSpecificPrediction(object):
    """
    Make sequence specific preditions with maxATAC.
    
    Args:
        signal (str): Path to ATAC-seq signal bigwig
        sequence (str): Path to 2bit DNA sequence
        rows (int, optional): Number of rows == channels. Defaults to INPUT_CHANNELS.
        cols (int, optional): Number of cols == region length. Defaults to INPUT_LENGTH.
        BP_ORDER (list, optional): Nucleotide order of one-hot encoding. Defaults to BP_ORDER.
          
    Example Usage:
    
    >>> AD4 = SequenceSpecificPrediction(model = CTCF.h5, signal = AD4_ATACseq.bigwig, sequence = hg38.2bit)
    >>> AD_A_predictions = AD4.predict(chromosome='chr1', variant_start=2565435, target_nucleotide="A")
    """
    def __init__(self,
                 signal: str,
                 sequence : str,
                 rows: int = INPUT_CHANNELS,
                 cols : int = INPUT_LENGTH,
                 BP_ORDER:list = BP_ORDER
                 ):
        """Initialize parameters

        Args:
            signal (str): Path to ATAC-seq signal bigwig
            sequence (str): Path to 2bit DNA sequence
            rows (int, optional): Number of rows == channels. Defaults to INPUT_CHANNELS.
            cols (int, optional): Number of cols == region length. Defaults to INPUT_LENGTH.
            BP_ORDER (list, optional): Nucleotide order of one-hot encoding. Defaults to BP_ORDER.
        """
        self.signal = signal
        self.sequence = sequence
        self.rows = rows
        self.cols = cols
        self.bp_order = BP_ORDER
        self.window = int(INPUT_LENGTH/2)
        self.BP_DICT = {"A":0, "C":1, "G":2, "T":3}

    def __str__(self):
        return f"Input bigwig: {self.signal} \n" + f"Input reference: {self.sequence} \n" 
    
    def __repr__(self):
        return f"Input bigwig: {self.signal} \n" + f"Input reference: {self.sequence} \n"
                
    def _create_sliding_windows_(self,
                                 chromosome: str, 
                                 variant_start_pos: int, 
                                 overhang: int = 256, 
                                 step_size: int = 256):
        """Create sliding windows around the variant position
        
        This function will create sliding windows around the variant psoition. This is a naive appoach that does not enforce
        chromosome boundaries. Use the overhang parameter to set the number of base pairs on each side to to predict in. This
        also assumes that the values will be in multiples of the window step.    
                
        Args:
            variant_start_pos (int): Start position of the variant
            overhang (int): Number of base pairs around the 1,024 bp region to include in prediction. Defaults to 256.
            step_size (int): Number of base pairs to slide the prediction window when an overhang is included. Use multiples of 32. Defaults to 256.
            
        Returns:
            list, dataframe: list of regions, dataframe of regions
        """
        logging.error(f"The predictions will be centered on position: {chromosome} {variant_start_pos} \n" + 
                      f"The central 1,024 bp window will have an overhang of {overhang} bp \n" + 
                      f"The windows will slide every {step_size} bp")
        
        # Use the overhang to calculate how many intervals will be created
        num_regions = int(((overhang / step_size) * 2) + 1)
        
        logging.error(f"There are {num_regions} total 1,024 bp windows that will make up the prediction.")
                     
        # Find the first windows center point
        window_center_index = variant_start_pos - overhang
        
        regions_list = []
        
        # create intervals 
        for _ in range(0 , num_regions):
            window_start = window_center_index - 512
            window_stop = window_center_index + 512
        
            regions_list.append([window_start, window_stop])
            
            window_center_index += step_size
        
        # Create a dataframe from the list
        windows_df = pd.DataFrame(regions_list, columns=["start", "stop"])

        windows_df["chr"] = chromosome
        
        return regions_list, windows_df[["chr", "start", "stop"]]

    def _get_seq_specific_input_matrix_(self, 
                                        chromosome: str, 
                                        window_start: int, 
                                        window_stop: int, 
                                        variant_index: int, 
                                        target_nucleotide:str):
        """Generate a variant specific input matrix for maxATAC

        Args:
            chromosome (str): The window chromosome
            window_start (int): The window start
            window_stop (int): The window stop
            variant_index (int): The index position in the array that corresponds to the variant position
            target_nucleotide (str): The target nucleotide that you want in the index position

        Returns:
            array: An np.array with dimensions 1024 x 5
        """
        # The nucleotide positions are 0-4
        signal_stream = load_bigwig(self.signal)
        sequence_stream  = load_2bit(self.sequence)
        
        input_matrix = np.zeros((self.rows, self.cols))
        
        for n, bp in enumerate(self.bp_order):
            # Get the sequence from the interval of interest
            target_sequence = Seq(sequence_stream.sequence(chromosome, window_start, window_stop))

            # Get the one hot encoded sequence
            input_matrix[n, :] = get_one_hot_encoded(target_sequence, bp)

        signal_array = np.array(signal_stream.values(chromosome, window_start, window_stop))

        input_matrix[4, :] = signal_array

        input_matrix = input_matrix.T
                
        if variant_index >= 0 and variant_index < 1024:
            # Get the nucleotide index
            nucleotide_index = self.BP_DICT[target_nucleotide]
        
            input_matrix[variant_index, nucleotide_index] = 1

            other_nucleotides = [item for item in [0,1,2,3] if item not in [nucleotide_index]]
            
            input_matrix[variant_index, other_nucleotides[0]] = 0
            input_matrix[variant_index, other_nucleotides[1]] = 0
            input_matrix[variant_index, other_nucleotides[2]] = 0
        else:
            pass
        
        return np.array([input_matrix])

    def _convert_predictions_to_bedgraph_(self, 
                                          predictions, 
                                          predict_roi_df, 
                                          chromosome):
        """Convert a arrays of maxATAC predictions to a bedgraph formatted dataframe

        Args:
            predictions ([type]): [description]
            predict_roi_df ([type]): [description]
            chromosome ([type]): [description]

        Returns:
            [type]: [description]
        """
        predictions_df = pd.DataFrame(data=predictions, index=None)
        
        predictions_df["chr"] = chromosome
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
    
    def _get_atac_signal_(self, chromosome):
        signal_stream = load_bigwig(self.signal)
        
        self.atac_min_coordinate = self.windows_df["start"].min()
        
        self.atac_max_coordinate = self.windows_df["stop"].max()
        
        self.atac_signal = np.array(signal_stream.values(chromosome, self.atac_min_coordinate, self.atac_max_coordinate))

    def predict(self, 
                model:str, 
                chromosome: str, 
                variant_start: int, 
                target_nucleotide: str, 
                overhang: int = 256, 
                step_size: int = 256):
        """Make a prediction around a specific genetic loci

        Args:
            model (str): Path to the maxATAC model
            chromosome (str): The chromosome to make predictions on
            variant_start (int): The position of the variant in 0-based coordinates
            target_nucleotide (str): The target nucleotide you which to have at the variant position
            overhang (int): The number of basepairs that you want to predict beyond the 1,024 bp prediction windows: similar to slop
            step_size (int): The number of base pairs that you want to slide each prediction window

        Returns:
            dataframe: A dataframe with bedgraph format
        """
        # Load the maxATAC model
        nn_model = load_model(model, compile=False)
  
        # Create the prediction windows
        self.prediction_windows, self.windows_df = self._create_sliding_windows_(chromosome=chromosome,
                                                                                 variant_start_pos=variant_start,
                                                                                 overhang=overhang,
                                                                                 step_size=step_size)
        
        prediction_list = []
        
        # We loop through the list of regions. The list of regions starts from the left most position.
        # We do not need exact genomic position, but the 
        # relative position of the variant compared to the 1,024 bp array that we are inputting into our model. 
        variant_index = 512 + overhang
        
        for window in self.prediction_windows:
            # Get the sequence specific input matrix for the genomic region of interest
            seq_specific_array = self._get_seq_specific_input_matrix_(chromosome=chromosome, 
                                                                      window_start=window[0], 
                                                                      window_stop=window[1], 
                                                                      variant_index=variant_index, 
                                                                      target_nucleotide=target_nucleotide)
            
            # Append the batch of predictions from the model
            prediction_list.append(nn_model.predict_on_batch(seq_specific_array).flatten())
            
            # subtract the set size from the variant index for the next windows calculation
            variant_index -= step_size
        
        # convert all predictions to a bedgraph format dataframe
        bedgraph_df = self._convert_predictions_to_bedgraph_(predictions=prediction_list, 
                                                             predict_roi_df = self.windows_df, 
                                                             chromosome=chromosome)
  
        self._get_atac_signal_(chromosome)
        
        return bedgraph_df
        