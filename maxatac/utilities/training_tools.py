import random
import pandas as pd
import numpy as np

from maxatac.utilities.genome_tools import (build_chrom_sizes_dict,
                                            get_input_matrix,
                                            import_bed,
                                            load_bigwig,
                                            load_2bit,
                                            get_target_matrix)


class GenomicDataGenerator(object):
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
                 sequence,
                 batch_size,
                 region_length,
                 peak_paths,
                 random_ratio,
                 chromosome_sizes,
                 bp_resolution,
                 input_channels,
                 cell_types,
                 scale_signal
                 ):
        self.meta_dataframe = meta_dataframe
        self.chromosomes = chromosomes
        self.blacklist = blacklist
        self.region_length = region_length
        self.peak_paths = peak_paths
        self.random_ratio = random_ratio
        self.input_channels = input_channels
        self.bp_resolution = bp_resolution
        self.batch_size = batch_size
        self.sequence = sequence
        self.region_length = region_length
        self.meta_dataframe = meta_dataframe
        self.peak_paths = peak_paths
        self.cell_types = cell_types
        self.scale_signal = scale_signal

        # Calculate the number of ROI and Random regions needed based on batch size and random ratio desired
        self.number_roi = round(batch_size * (1. - random_ratio))
        self.number_random_regions = round(batch_size - self.number_roi)

        self.chromosome_sizes_dictionary = build_chrom_sizes_dict(chromosomes, chromosome_sizes)

        # Get the ROIPool and/or RandomRegionsPool
        self.ROI_pool = self.__get_ROIPool()

        self.RandomRegions_pool = self.__get_RandomRegionsPool()

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

    def batch_generator(self):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :return: A batch of training examples centered on regions of interest
        """
        while True:
            inputs_batch, targets_batch = [], []

            for region in self.__mix_regions():
                signal, binding = self.__get_random_cell_data()

                with load_2bit(self.sequence) as sequence_stream, load_bigwig(signal) as signal_stream, \
                        load_bigwig(binding) as binding_stream:
                    inputs_batch.append(get_input_matrix(rows=self.input_channels,
                                                         cols=self.region_length,
                                                         bp_order=["A", "C", "G", "T"],
                                                         signal_stream=signal_stream,
                                                         sequence_stream=sequence_stream,
                                                         chromosome=region[0],
                                                         start=region[1],
                                                         end=region[2],
                                                         scale_signal=self.scale_signal
                                                         )
                                        )

                    targets_batch.append(get_target_matrix(binding_stream,
                                                           chromosome=region[0],
                                                           start=region[1],
                                                           end=region[2],
                                                           bp_resolution=self.bp_resolution
                                                           )
                                         )

            yield np.array(inputs_batch), np.array(targets_batch)


class RandomRegionsPool(object):
    """
    This class will generate a pool of random regions

    The RandomRegionsGenerator will generate a random region of interest based on the input reference genome
    chromosome sizes, the desired size of the chromosome pool, and the input length.
    """

    def __init__(
            self,
            chromosome_sizes_dictionary,
            region_length,
            meta_dataframe
    ):
        """
        :param chromosome_sizes_dictionary: Dictionary of chromosome sizes filtered for chromosomes of interest
        :param region_length: Length of the training regions to be used
        :param meta_dataframe: Meta table used to ID location of peaks and signals
        """
        self.chromosome_sizes_dictionary = chromosome_sizes_dictionary
        self.region_length = region_length
        self.__idx = 0
        self.meta_dataframe = meta_dataframe

    def __get_random_region(self):
        """
        Get a random region from the pool.

        :return: The chromosome string, region start, and region end
        """
        chrom_name, chrom_length = random.choice(list(self.chromosome_sizes_dictionary.items()))

        start = round(random.randint(0, chrom_length - self.region_length))

        end = start + self.region_length

        return [chrom_name, start, end]

    def get_regions_list(self, number_random_regions):
        """
        Create batches of examples with the size of number_random_regions

        :param number_random_regions: Number of random regions to generate per batch

        :return: Training examples generated from random regions of the genome
        """
        random_regions_list = []

        for idx in range(number_random_regions):
            random_regions_list.append(self.__get_random_region())

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