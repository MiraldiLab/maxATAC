import random
import pandas as pd
import numpy as np
import keras

from maxatac.utilities.genome_tools import (build_chrom_sizes_dict,
                                            get_input_matrix,
                                            import_bed,
                                            load_bigwig,
                                            load_2bit,
                                            get_target_matrix)


class DataGenerator(keras.utils.Sequence):
    """
    Generate batches of examples for the chromosomes of interest

    This generator will create individual batches of examples from a pool of regions of interest and/or a pool of
    random regions from the genome. This class object helps keep track of all of the required inputs and how they are
    processed.

    This generator expects a meta_table with the header:

    TF | Cell Type | ATAC signal | ChIP signal | ATAC peaks | ChIP peaks
    """

    def __init__(self,
                 meta_dataframe,
                 chromosomes,
                 blacklist,
                 region_length,
                 average,
                 sequence,
                 batch_size,
                 bp_resolution,
                 input_channels,
                 chromosome_pool_size,
                 chromosome_sizes,
                 random_ratio,
                 peak_paths,
                 cell_types,
                 batches_per_epoch,
                 shuffle=True):
        """
        :param meta_dataframe: Path to meta table
        :param chromosomes: List of chromosome to restrict data to
        :param blacklist: Path to blacklist BED regions to exclude
        :param region_length: Length of the inputs to use
        :param average: Path to the average signal track
        :param sequence: Path to the 2bit DNA sequence
        :param batch_size: Number of examples to generate per batch
        :param bp_resolution: Resolution of the output prediction
        :param input_channels: Number of input channels
        :param chromosome_pool_size: Size of the chromosome pool used for generating random regions
        :param chromosome_sizes: A txt file of chromosome sizes
        :param random_ratio: The proportion of the total batch size that will be from randomly generated examples
        :param batches_per_epoch: The number of batches to use per epoch
        """
        self.chromosome_pool_size = chromosome_pool_size
        self.input_channels = input_channels
        self.bp_resolution = bp_resolution
        self.batch_size = batch_size
        self.sequence = sequence
        self.average = average
        self.region_length = region_length
        self.blacklist = blacklist
        self.chromosomes = chromosomes
        self.meta_dataframe = meta_dataframe
        self.random_ratio = random_ratio
        self.peak_paths = peak_paths
        self.cell_types = cell_types
        self.batch_per_epoch = batches_per_epoch
        self.shuffle = shuffle

        # Calculate the number of ROI and Random regions needed based on batch size and random ratio desired
        self.number_roi = round(batch_size * (1. - random_ratio))
        self.number_random_regions = round(batch_size - self.number_roi)

        self.chromosome_sizes_dictionary = build_chrom_sizes_dict(chromosomes, chromosome_sizes)

        # Get the ROIPool and/or RandomRegionsPool
        if random_ratio < 1:
            self.ROI_pool = self.__get_ROIPool()

        if random_ratio > 0:
            self.RandomRegions_pool = self.__get_RandomRegionsPool()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.batch_per_epoch

    def __getitem__(self, index):
        """Generate one batch of data"""
        current_batch = self.__mix_regions()

        # Generate data
        X, y = self.__data_generation(current_batch)

        return X, y

    def __get_ROIPool(self):
        """
        Passes the attributes to the ROIPool class to build a pool of regions of interest

        :return: Initializes the object used to generate batches of peak centered training examples.
        """
        return ROIPool(meta_dataframe=self.meta_dataframe,
                       chromosomes=self.chromosomes,
                       chromosome_sizes_dictionary=self.chromosome_sizes_dictionary,
                       blacklist=self.blacklist,
                       region_length=self.region_length,
                       peak_paths=self.peak_paths
                       )

    def __get_RandomRegionsPool(self):
        """
        Passes the attributes to the RandomRegionsPool class to build a pool of randomly generated training examples

        :return: Initializes the object used to generate batches of randomly generated examples
        """
        return RandomRegionsPool(chromosome_sizes_dictionary=self.chromosome_sizes_dictionary,
                                 chromosome_pool_size=self.chromosome_pool_size,
                                 region_length=self.region_length,
                                 meta_dataframe=self.meta_dataframe
                                 )

    def __mix_regions(self):
        """
        A generator that will combine batches of examples from two different sources with the defined proportions

        :return: Yields batches of training examples.
        """
        # Build a batch of examples with mixed random and ROI examples
        if 0. < self.random_ratio < 1.:

            # Initialize the random and ROI generators with the specified batch size based on the total batch size
            random_examples_list = self.RandomRegions_pool.get_regions_list(
                number_random_regions=self.number_random_regions)
            roi_examples_list = self.ROI_pool.get_regions_list(n_roi=self.number_roi)

            # Mix batches
            regions_list = random_examples_list + roi_examples_list

        # Build a batch of examples with only random examples
        elif self.random_ratio == 1.:
            # Initialize the RandomRegions generator only with the specified batch size
            regions_list = self.RandomRegions_pool.get_regions_list(number_random_regions=self.number_random_regions)

        # Build a batch of examples with only ROI examples
        else:
            # Initialize the ROI generator only with the specified batch size
            regions_list = self.ROI_pool.get_regions_list(n_roi=self.number_roi)

        return regions_list

    def __get_random_cell_data(self):
        """
        Get the cell line data from a random cell line from the meta table

        :return: returns the signal file and the binding data file from a random cell line
        """
        meta_row = self.meta_dataframe[self.meta_dataframe['Cell_Line'] == random.choice(self.cell_types)].reset_index(
            drop=True)

        return meta_row.loc[0, 'ATAC_Signal_File'], meta_row.loc[0, 'Binding_File']

    def __data_generation(self, current_batch):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :param current_batch: Current batch of regions

        :return: A batch of training examples centered on regions of interest
        """
        while True:
            inputs_batch, targets_batch = [], []

            for region in current_batch:
                signal, binding = self.__get_random_cell_data()

                with \
                        load_bigwig(self.average) as average_stream, \
                        load_2bit(self.sequence) as sequence_stream, \
                        load_bigwig(signal) as signal_stream, \
                        load_bigwig(binding) as binding_stream:
                    inputs_batch.append(get_input_matrix(rows=self.input_channels,
                                                         cols=self.region_length,
                                                         bp_order=["A", "C", "G", "T"],
                                                         signal_stream=signal_stream,
                                                         average_stream=average_stream,
                                                         sequence_stream=sequence_stream,
                                                         chromosome=region[0],
                                                         start=region[1],
                                                         end=region[2]
                                                         )
                                        )

                    targets_batch.append(get_target_matrix(binding_stream,
                                                           chromosome=region[0],
                                                           start=region[1],
                                                           end=region[2],
                                                           bp_resolution=self.bp_resolution
                                                           )
                                         )

            return np.array(inputs_batch), np.array(targets_batch)


class RandomRegionsPool(object):
    """
    This class will generate a pool of random regions

    The RandomRegionsGenerator will generate a random region of interest based on the input reference genome
    chromosome sizes, the desired size of the chromosome pool, the input length, and the method used to calculate the
    frequency of chromosome examples.
    """

    def __init__(
            self,
            chromosome_sizes_dictionary,
            chromosome_pool_size,
            region_length,
            meta_dataframe,
            method="length"
    ):
        """
        :param chromosome_sizes_dictionary: Dictionary of chromosome sizes filtered for chromosomes of interest
        :param chromosome_pool_size: Size of the pool to use for building the pool to sample from
        :param region_length: Length of the training regions to be used
        :param meta_dataframe: Meta table used to ID location of peaks and signals
        :param method: Method to use to build the random regions pool
        :param cell_types: List of cell types available
        """
        self.chromosome_sizes_dictionary = chromosome_sizes_dictionary
        self.chromosome_pool_size = chromosome_pool_size
        self.region_length = region_length
        self.__idx = 0
        self.method = method
        self.meta_dataframe = meta_dataframe

        self.chromosome_pool = self.__get_chromosome_frequencies()

    def __get_chromosome_frequencies(self):
        """
        Generate an array of chromosome frequencies

        The frequencies will be used to build the pool of regions that we pull random regions from. The frequencies
        are determined by the method attribute. The methods are "length" or "proportion".

        The length method will generate the frequencies of examples in the pool based on the length of the chromosomes
        in total.

        The proportion method will generate a pool that has chromosome frequencies equal to the chromosome pools size
        divided by the number of chromosomes.

        :return:
        """
        if self.method == "length":
            sum_lengths = sum(self.chromosome_sizes_dictionary.values())

            frequencies = {
                chromosome: round(chromosome_len / sum_lengths * self.chromosome_pool_size)
                for chromosome, chromosome_len in self.chromosome_sizes_dictionary.items()
            }

        else:
            chromosome_number = len(self.chromosome_sizes_dictionary.values())

            frequencies = {
                chromosome: round(self.chromosome_pool_size / chromosome_number)
                for chromosome, chromosome_len in self.chromosome_sizes_dictionary.items()
            }

        labels = []

        for k, v in frequencies.items():
            labels += [(k, self.chromosome_sizes_dictionary[k])] * v
        random.shuffle(labels)

        return labels

    def __get_region(self):
        """
        Get a random region from the pool.

        :return: The chromosome string, region start, and region end
        """

        # If the __idx reaches the pool size before enough samples are selected
        # shuffle and set to 0
        if self.__idx == self.chromosome_pool_size:
            random.shuffle(self.chromosome_pool)
            self.__idx = 0

        chrom_name, chrom_length = self.chromosome_pool[self.__idx]
        self.__idx += 1

        start = round(
            random.randint(
                0,
                chrom_length - self.region_length
            )
        )

        end = start + self.region_length

        return chrom_name, start, end

    def get_regions_list(self, number_random_regions):
        """
        Create batches of examples with the size of number_random_regions

        :param number_random_regions: Number of random regions to generate per batch

        :return: Training examples generated from random regions of the genome
        """
        random_regions_list = []

        for idx in range(number_random_regions):
            random_regions_list.append(self.__get_region())

        return random_regions_list


class ROIPool(object):
    """
    This class will generate a pool of examples based on regions of interest defined by ATAC-seq and ChIP-seq peaks

    The RandomRegionsGenerator will generate a random region of interest based on the input reference genome
    chromosome sizes, the desired size of the chromosome pool, the input length, and the method used to calculate the
    frequency of chromosome examples.
    """

    def __init__(self,
                 meta_dataframe,
                 chromosomes,
                 chromosome_sizes_dictionary,
                 blacklist,
                 region_length,
                 peak_paths
                 ):
        """
        :param meta_dataframe: Path to the meta file
        :param chromosomes: List of chromosomes to use
        :param chromosome_sizes_dictionary: A dictionary of chromosome sizes
        :param blacklist: The blacklist file of BED regions to exclude
        :param peak_paths: List of paths to ATAC and ChIP-seq peaks
        :param region_length: Length of the input regions
        """
        self.meta_dataframe = meta_dataframe
        self.chromosome_sizes_dictionary = chromosome_sizes_dictionary
        self.chromosomes = chromosomes
        self.blacklist = blacklist
        self.region_length = region_length
        self.peak_paths = peak_paths

        self.roi_pool = self.__get_roi_pool()

        self.roi_size = self.roi_pool.shape[0]

    def __get_roi_pool(self):
        """
        Build a ROI pool from all of the peak files of interest

        :return: A dataframe of BED regions
        """
        bed_list = []

        for bed_file in self.peak_paths:
            bed_list.append(import_bed(bed_file,
                                       region_length=self.region_length,
                                       chromosomes=self.chromosomes,
                                       chromosome_sizes_dictionary=self.chromosome_sizes_dictionary,
                                       blacklist=self.blacklist))

        return pd.concat(bed_list).sample(frac=1)

    def get_regions_list(self,
                         n_roi):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :param n_roi: Number of regions to generate per batch

        :return: A batch of training examples centered on regions of interest
        """
        random_roi_pool = self.roi_pool.sample(n=n_roi, replace=True, random_state=1)

        return random_roi_pool.to_numpy().tolist()
