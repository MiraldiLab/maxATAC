import random
import pandas as pd
import numpy as np

from maxatac.utilities.genome_tools import build_chrom_sizes_dict, get_input_matrix, import_bed, load_bigwig, load_2bit


class DataGenerator(object):
    """
    Generate batches of examples for the chromosomes of interest

    This generator will create individual batches of examples from a pool of regions of interest and/or a pool of
    random regions from the genome. This class object helps keep track of all of the required inputs and how they are
    processed.

    The random_ratio attribute controls the ratio of random and ROI examples.

    This generator expects a meta_table with the header:

    TF | Cell Type | ATAC signal | ChIP signal | ATAC peaks | ChIP peaks

    The method _get_ROIPool method will retrieve the ROI_pool (pool of regions of interest examples)

    The method _get_RandomRegionsPool will get a batch of random examples from the genome

    The method BatchGenerator is a generator that will yield training examples based on the attributes above.

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
                 cell_types):
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
        self.rand_ratio = random_ratio
        self.peak_paths = peak_paths
        self.cell_types = cell_types

        # Calculate the number of ROI and Random regions needed based on batch size and random ratio desired
        self.number_roi = round(batch_size * (1. - random_ratio))
        self.number_random_regions = round(batch_size - self.number_roi)

        self.chromosome_sizes_dictionary = build_chrom_sizes_dict(chromosomes, chromosome_sizes)

        # Get the ROIPool and/or RandomRegionsPool
        if random_ratio < 1:
            self.ROI_pool = self.__get_ROIPool()

        if random_ratio > 0:
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
                       average=self.average,
                       sequence=self.sequence,
                       batch_size=self.batch_size,
                       bp_resolution=self.bp_resolution,
                       input_channels=self.input_channels,
                       cell_types=self.cell_types,
                       peak_paths=self.peak_paths)

    def __get_RandomRegionsPool(self):
        """
        Passes the attributes to the RandomRegionsPool class to build a pool of randomly generated training examples

        :return: Initializes the object used to generate batches of randomly generated examples
        """
        return RandomRegionsPool(chromosome_sizes_dictionary=self.chromosome_sizes_dictionary,
                                 chromosome_pool_size=self.chromosome_pool_size,
                                 region_length=self.region_length,
                                 sequence=self.sequence,
                                 average=self.average,
                                 meta_dataframe=self.meta_dataframe,
                                 input_channels=self.input_channels,
                                 bp_resolution=self.bp_resolution,
                                 cell_types=self.cell_types)

    def BatchGenerator(self):
        """
        A generator that will combine batches of examples from two different sources with the defined proportions

        :return: Yields batches of training examples.
        """
        while True:
            # Build a batch of examples with mixed random and ROI examples
            if 0. < self.rand_ratio < 1.:

                # Initialize the random and ROI generators with the specified batch size based on the total batch size
                rand_gen = self.RandomRegions_pool.generate_batch(number_random_regions=self.number_random_regions)
                roi_gen = self.ROI_pool.generate_batch(n_roi=self.number_roi)

                # Yield the batches of examples
                roi_input_batch, roi_target_batch = next(roi_gen)
                rand_input_batch, rand_target_batch = next(rand_gen)

                # Mix batches
                inputs_batch = np.concatenate((roi_input_batch, rand_input_batch), axis=0)
                targets_batch = np.concatenate((roi_target_batch, rand_target_batch), axis=0)

            # Build a batch of examples with only random examples
            elif self.rand_ratio == 1.:
                # Initialize the RandomRegions generator only with the specified batch size
                rand_gen = self.RandomRegions_pool.generate_batch(number_random_regions=self.number_random_regions)

                # Yield the batch of random examples
                inputs_batch, targets_batch = next(rand_gen)

            # Build a batch of examples with only ROI examples
            else:
                # Initialize the ROI generator only with the specified batch size
                roi_gen = self.ROI_pool.generate_batch(n_roi=self.number_roi)

                # Yield the batch of ROI examples
                inputs_batch, targets_batch = next(roi_gen)

            yield inputs_batch, targets_batch


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
            cell_types,
            sequence,
            average,
            meta_dataframe,
            input_channels,
            bp_resolution,
            method="length"
    ):
        """
        :param chromosome_sizes_dictionary: Dictionary of chromosome sizes filtered for chromosomes of interest
        :param chromosome_pool_size: Size of the pool to use for building the pool to sample from
        :param region_length: Length of the training regions to be used
        :param sequence: 2bit DNA sequence
        :param average: Average ATAC-seq signal file
        :param meta_dataframe: Meta table used to ID location of peaks and signals
        :param input_channels: The number of input channels to use
        :param bp_resolution: Resolution of the output predictions
        :param method: Method to use to build the random regions pool
        :param cell_types: List of cell types available
        """
        self.chromosome_sizes_dictionary = chromosome_sizes_dictionary
        self.chromosome_pool_size = chromosome_pool_size
        self.region_length = region_length
        self.__idx = 0
        self.method = method
        self.sequence = sequence
        self.average = average
        self.meta_dataframe = meta_dataframe
        self.input_channels = input_channels
        self.bp_resolution = bp_resolution
        self.cell_types = cell_types

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

    def __get_random_cell_data(self):
        """
        Get the cell line data from a random cell line from the meta table

        :return: returns the signal file and the binding data file from a random cell line
        """
        meta_row = self.meta_dataframe[self.meta_dataframe['Cell_Line'] == random.choice(self.cell_types)].reset_index(
            drop=True)

        return meta_row.loc[0, 'ATAC_Signal_File'], meta_row.loc[0, 'Binding_File']

    def generate_batch(self, number_random_regions):
        """
        Create batches of examples with the size of number_random_regions

        :param number_random_regions: Number of random regions to generate per batch

        :return: Training examples generated from random regions of the genome
        """
        while True:
            inputs_batch, targets_batch = [], []

            for idx in range(number_random_regions):
                chromosome, start, end = self.__get_region()  # returns random region (chromosome, start, end)

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
                                                         chromosome=chromosome,
                                                         start=start,
                                                         end=end
                                                         )
                                        )

                    targets_batch.append(get_target_matrix(binding_stream,
                                                           chromosome=chromosome,
                                                           start=start,
                                                           end=end,
                                                           bp_resolution=self.bp_resolution))

            yield np.array(inputs_batch), np.array(targets_batch)


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
                 average,
                 sequence,
                 batch_size,
                 bp_resolution,
                 input_channels,
                 cell_types,
                 peak_paths
                 ):
        """
        :param meta_dataframe: Path to the meta file
        :param chromosomes: List of chromosomes to use
        :param chromosome_sizes_dictionary: A dictionary of chromosome sizes
        :param blacklist: The blacklist file of BED regions to exclude
        :param region_length: Length of the input regions
        :param average: Average signal track
        :param sequence: 2Bit DNA sequence
        :param batch_size: Number of examples per batch generated
        :param bp_resolution: Prediction resolution in base pairs
        :param input_channels: Number of input channels
        """
        self.meta_dataframe = meta_dataframe
        self.chromosome_sizes_dictionary = chromosome_sizes_dictionary
        self.chromosomes = chromosomes
        self.blacklist = blacklist
        self.region_length = region_length
        self.average = average
        self.sequence = sequence
        self.batch_size = batch_size
        self.bp_resolution = bp_resolution
        self.input_channels = input_channels
        self.cell_types = cell_types
        self.peak_paths = peak_paths

        self.roi_pool = self.__get_roi_pool()

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

    def generate_batch(self,
                       n_roi):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :param n_roi: Number of regions to generate per batch

        :return: A batch of training examples centered on regions of interest
        """
        while True:
            inputs_batch, targets_batch = [], []

            roi_size = self.roi_pool.shape[0]

            curr_batch_idxs = random.sample(range(roi_size), n_roi)

            # Here I will process by row, if performance is bad then process by cell line
            for row_idx in curr_batch_idxs:
                roi_row = self.roi_pool.iloc[row_idx, :]

                cell_line = random.sample(self.cell_types, 1)[0]

                chromosome = roi_row['chr']

                start = int(roi_row['start'])

                end = int(roi_row['stop'])

                meta_row = self.meta_dataframe[self.meta_dataframe['Cell_Line'] == cell_line]

                meta_row = meta_row.reset_index(drop=True)

                signal = meta_row.loc[0, 'ATAC_Signal_File']

                binding = meta_row.loc[0, 'Binding_File']

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
                                                         chromosome=chromosome,
                                                         start=start,
                                                         end=end
                                                         )
                                        )

                    targets_batch.append(get_target_matrix(binding_stream,
                                                           chromosome=chromosome,
                                                           start=start,
                                                           end=end,
                                                           bp_resolution=self.bp_resolution
                                                           )
                                         )

            yield np.array(inputs_batch), np.array(targets_batch)


def get_target_matrix(binding,
                      chromosome,
                      start,
                      end,
                      bp_resolution):
    """
    Get the values from a ChIP-seq signal file

    :param binding: ChIP-seq signal file
    :param chromosome: chromosome name: chr1
    :param start: start
    :param end: end
    :param bp_resolution: Prediction resolution in base pairs

    :return: Returns a vector of binary values
    """
    target_vector = np.array(binding.values(chromosome, start, end)).T

    target_vector = np.nan_to_num(target_vector, 0.0)

    n_bins = int(target_vector.shape[0] / bp_resolution)

    split_targets = np.array(np.split(target_vector, n_bins, axis=0))

    bin_sums = np.sum(split_targets, axis=1)

    return np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)
