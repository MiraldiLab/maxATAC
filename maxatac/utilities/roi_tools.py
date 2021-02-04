import os
import random
import pandas as pd
import numpy as np
from joblib._multiprocessing_helpers import mp

from maxatac.utilities.genome_tools import (build_chrom_sizes_dict,
                                            import_bed)

from maxatac.utilities.system_tools import get_dir


class ValidationData(object):
    """
    Generate validation data based on ATAC-seq and ChIP-seq peaks from the run meta file.

    This generator will create individual batches of examples from a pool of regions of interest and/or a pool of
    random regions from the genome. This class object helps keep track of all of the required inputs and how they are
    processed.

    This generator expects a meta_table with the header:

    TF | Cell_Type | ATAC_Signal_File | Binding_File | ATAC_Peaks | ChIP_peaks
    """

    def __init__(self,
                 meta_path,
                 chromosomes,
                 blacklist,
                 region_length,
                 random_ratio,
                 chromosome_sizes,
                 preferences,
                 threads
                 ):
        """
        :param meta_path: Path to run meta file
        :param chromosomes: List of chromosomes to limit the pool to
        :param blacklist: BED file of blacklisted regions to remove
        :param region_length: Length of regions to produce
        :param random_ratio: The degree of random ratios to add to batch
        :param chromosome_sizes: A file of chromosome sizes
        :param preferences: The BED file of blacklist complement regions
        """
        self.meta_path = meta_path
        self.chromosomes = chromosomes
        self.blacklist = blacklist
        self.region_length = region_length
        self.random_ratio = random_ratio
        self.region_length = region_length
        self.preferences = preferences
        self.threads = threads
        self.chromosome_sizes_dictionary = build_chrom_sizes_dict(chromosomes, chromosome_sizes)

        # Import meta txt as dataframe
        self.meta_dataframe = pd.read_csv(self.meta_path, sep='\t', header=0, index_col=None)
        self.cell_types = self.meta_dataframe["Cell_Line"].unique()

        # Get the ROIPool and/or RandomRegionsPool
        self.ROI_pool = self.__get_ROIPool()
        self.number_roi = self.ROI_pool.combined_pool.shape[0]

        self.number_random_regions = int((self.number_roi/(1 - self.random_ratio)) - self.number_roi)
        self.RandomRegions_pool = self.__get_RandomRegionsPool()

        self.validation_regions_pool = self.__get_validation_pool()

    def __get_ROIPool(self):
        """
        Passes the attributes to the ROIPool class to build a pool of regions of interest

        :return: Initializes the object used to generate batches of peak centered training examples.
        """
        return ROIPool(meta_path=self.meta_path,
                       chromosomes=self.chromosomes,
                       chromosome_sizes_dictionary=self.chromosome_sizes_dictionary,
                       blacklist=self.blacklist,
                       region_length=self.region_length
                       )

    def __get_RandomRegionsPool(self):
        """
        Passes the attributes to the RandomRegionsPool class to build a pool of randomly generated training examples

        :return: Initializes the object used to generate batches of randomly generated examples
        """
        return RandomRegionsPool(chromosomes=self.chromosomes,
                                 chromosome_sizes_dictionary=self.chromosome_sizes_dictionary,
                                 region_length=self.region_length,
                                 preferences=self.preferences,
                                 cell_types=self.cell_types,
                                 threads=self.threads)

    def __get_validation_pool(self):
        """
        Combine batches of examples from two different sources with the defined proportions

        :return: Yields batches of training examples.
        """
        # Build a batch of examples with mixed random and ROI examples
        # Initialize the random and ROI generators with the specified batch size based on the total batch size
        random_examples_list = self.RandomRegions_pool.get_regions_list(
                number_random_regions=self.number_random_regions)

        roi_examples_list = self.ROI_pool.get_regions_list(n_roi=self.number_roi)

        # Mix batches
        regions_list = random_examples_list + roi_examples_list

        validation_dataframe = pd.DataFrame(regions_list, columns=["Chr", "Start", "Stop", "ROI_Type", "Cell_Line"])

        return validation_dataframe

    def write_validation_pool(self, prefix="test", output_dir="./ROI"):
        """
        Write the pool dataframe into a TSV, BED and a summary stats file.

        :param prefix:
        :param output_dir:
        :return:
        """
        output_directory = get_dir(output_dir)
        bed_filename = os.path.join(output_directory, prefix + "_validation_ROI.bed")
        tsv_filename = os.path.join(output_directory, prefix + "_validation_ROI.tsv")

        stats_filename = os.path.join(output_directory, prefix + "_validation_ROI_stats.tsv")
        total_regions_filename = os.path.join(output_directory, prefix + "_validation_ROI_total_regions_stats.tsv")

        self.validation_regions_pool.to_csv(bed_filename, sep="\t", index=False, header=False)
        self.validation_regions_pool.to_csv(tsv_filename, sep="\t", index=False)

        group_ms = self.validation_regions_pool.groupby(["Chr", "Cell_Line", "ROI_Type"], as_index=False).size()
        len_ms = self.validation_regions_pool.shape[0]

        group_ms.to_csv(stats_filename, sep="\t", index=False)

        file = open(total_regions_filename, "a")
        file.write('Total number of regions found for training are: {0}\n'.format(len_ms))
        file.close()


class ROIPool(object):
    """
    This class will generate a pool of examples based on regions of interest defined by ATAC-seq and ChIP-seq peaks.
    """

    def __init__(self,
                 meta_path,
                 chromosomes,
                 chromosome_sizes_dictionary,
                 blacklist,
                 region_length
                 ):
        """
        When the object is initialized it will import all of the peaks in the meta files and parse them into training
        and validation regions of interest. these will be in the form of TSV formatted file similar to a BED file.

        :param meta_path: Path to the meta file
        :param chromosomes: List of chromosomes to use
        :param chromosome_sizes_dictionary: A dictionary of chromosome sizes
        :param blacklist: The blacklist file of BED regions to exclude
        :param region_length: Length of the input regions
        """
        self.meta_path = meta_path
        self.chromosome_sizes_dictionary = chromosome_sizes_dictionary
        self.chromosomes = chromosomes
        self.blacklist = blacklist
        self.region_length = region_length

        # Import meta txt as dataframe
        self.meta_dataframe = pd.read_csv(self.meta_path, sep='\t', header=0, index_col=None)

        self.cell_types = self.meta_dataframe["Cell_Line"].unique()

        # Get a dictionary of {Cell Types: Peak Paths}
        self.atac_dictionary = pd.Series(self.meta_dataframe.ATAC_Peaks.values,
                                         index=self.meta_dataframe.Cell_Line).to_dict()

        self.chip_dictionary = pd.Series(self.meta_dataframe.CHIP_Peaks.values,
                                         index=self.meta_dataframe.Cell_Line).to_dict()

        # You must generate the ROI pool before you can get the final shape
        self.atac_roi_pool = self.__get_roi_pool(self.atac_dictionary, "ATAC", )
        self.chip_roi_pool = self.__get_roi_pool(self.chip_dictionary, "CHIP")

        self.combined_pool = pd.concat([self.atac_roi_pool, self.chip_roi_pool])

        self.atac_roi_size = self.atac_roi_pool.shape[0]
        self.chip_roi_size = self.chip_roi_pool.shape[0]

    def __get_roi_pool(self, dictionary, roi_type_tag):
        """
        Build a pool of regions of interest from BED files.

        :param dictionary: A dictionary of Cell Types and their associated BED files
        :param roi_type_tag: Tag used to name the type of ROI being generated. IE Chip or ATAC

        :return: A dataframe of BED regions that are formatted for maxATAC training.
        """
        bed_list = []

        for roi_cell_tag, bed_file in dictionary.items():
            bed_list.append(import_bed(bed_file,
                                       region_length=self.region_length,
                                       chromosomes=self.chromosomes,
                                       chromosome_sizes_dictionary=self.chromosome_sizes_dictionary,
                                       blacklist=self.blacklist,
                                       ROI_type_tag=roi_type_tag,
                                       ROI_cell_tag=roi_cell_tag))

        return pd.concat(bed_list)

    def write_ROI_pools(self, prefix="ROI_pool", output_dir="./ROI"):
        """
        Write the ROI dataframe to a tsv and a bed for for ATAC, CHIP, and combined ROIs

        :param prefix: Prefix for filenames to use
        :param output_dir: Directory to output the bed and tsv files

        :return: Write BED and TSV versions of the ROI data
        """
        output_directory = get_dir(output_dir)

        atac_BED_filename = os.path.join(output_directory, prefix + "_training_ATAC_ROI.bed")
        chip_BED_filename = os.path.join(output_directory, prefix + "_training_CHIP_ROI.bed")
        combined_BED_filename = os.path.join(output_directory, prefix + "_training_ROI.bed")

        atac_TSV_filename = os.path.join(output_directory, prefix + "_training_ATAC_ROI.tsv")
        chip_TSV_filename = os.path.join(output_directory, prefix + "_training_CHIP_ROI.tsv")
        combined_TSV_filename = os.path.join(output_directory, prefix + "_training_ROI.tsv")

        stats_filename = os.path.join(output_directory, prefix + "_training_ROI_stats.tsv")
        total_regions_stats_filename = os.path.join(output_directory, prefix + "_training_ROI_totalregions_stats.tsv")

        self.atac_roi_pool.to_csv(atac_BED_filename, sep="\t", index=False, header=False)
        self.chip_roi_pool.to_csv(chip_BED_filename, sep="\t", index=False, header=False)
        self.combined_pool.to_csv(combined_BED_filename, sep="\t", index=False, header=False)
        self.atac_roi_pool.to_csv(atac_TSV_filename, sep="\t", index=False)
        self.chip_roi_pool.to_csv(chip_TSV_filename, sep="\t", index=False)
        self.combined_pool.to_csv(combined_TSV_filename, sep="\t", index=False)

        group_ms = self.combined_pool.groupby(["Chr", "Cell_Line", "ROI_Type"], as_index=False).size()
        len_ms = self.combined_pool.shape[0]
        group_ms.to_csv(stats_filename, sep="\t", index=False)

        file = open(total_regions_stats_filename, "a")
        file.write('Total number of regions found for training are: {0}\n'.format(len_ms))
        file.close()

    def get_regions_list(self,
                         n_roi):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :param n_roi: Number of regions to generate per batch

        :return: A batch of training examples centered on regions of interest
        """
        random_roi_pool = self.combined_pool.sample(n=n_roi, replace=True, random_state=1)

        return random_roi_pool.to_numpy().tolist()


class RandomRegionsPool(object):
    """
    This class will generate a pool of random regions

    The RandomRegionsGenerator will generate a random region of interest based on the input reference genome
    chromosome sizes, the desired size of the chromosome pool, and the input length.

    """

    def __init__(
            self,
            chromosomes,
            region_length,
            preferences,
            chromosome_sizes_dictionary,
            cell_types,
            threads
    ):
        """
        This object requires a dictionary of chromosome sizes that are filtered for the training chromosomes of
        interest. The chromosome sizes dictionary will be used to build the preference pool. The preference pool is a
        pool of chromosome regions that are appropriate for use in training. That means there are no blacklisted
        regions, telomeres, centromeres, or gaps.

        :param chromosomes: List of chromosomes to use for validation
        :param cell_types: List of cell types available
        :param region_length: Length of the training regions to be used
        :param preferences: The bed file of regions to limit the random selection to
        :param chromosome_sizes_dictionary: A dictionary of chromosome sizes filtered for the input chromosomes list
        """
        self.chromosomes = chromosomes
        self.region_length = region_length
        self.chromosome_sizes_dictionary = chromosome_sizes_dictionary
        self.cell_types = cell_types
        self.preferences_pool = self.__get_random_regions_pool(preferences)
        self.threads = threads
        self.__get_interval_weights()

    def __get_random_regions_pool(self, preferences):
        """
        This function will generate a dataframe of acceptable genomic regions to sample from.

        The intervals are divided by blacklist regions, gaps, telomeres and centromeres into unequal sized regions.
        We want to randomly choose an interval and then find a region within the interval. Using this simple approach
        means any interval has an equal chance of being chose. However, those smaller intervals will only produce a
        smaller pool of potential regions than larger regions. This means you have a have a larger probability of
        generating an interval that has already been chosen before.

        Example:
            choose 3 regions 5bp wide from a 10 bp interval:
              {     }
               {     }
            {     }
            |----------|

            compared to choosing 3 regions 5 bp wide from a 50 bp interval:
                                  {     }
                                                {     }
            {     }
            |-------------------------------------------------|

        Random regions from smaller intervals are over represented. The same concept is applied to chromosomes. Smaller
        chromosomes can be have more non-unique intervals than a larger chromosome sampled at the same frequency. This
        logic is similar to what is used by Leopard.
        """
        df = pd.read_csv(preferences,
                         sep="\t",
                         usecols=[0, 1, 2],
                         header=None,
                         names=["chr", "start", "stop"],
                         low_memory=False)

        # Make sure the chromosomes in the ROI file frame are in the target chromosome list
        df = df[df["chr"].isin(self.chromosomes)]

        # Find the region lengths for each interval. This information will be used to to generate the probabilities for
        # how frequently an interval will be chosen from. The intervals are divided by blacklist regions, gaps,
        # telomeres and centromeres into unqual sized regions. We want to sample randomly choose an interval with larger
        # regions have a larger probability of being chosen so regions from smaller intervals are not over represented.
        df["length"] = df["stop"] - df["start"]

        # We just want to simply filter out any intervals that are not at least 2X larger than the input regions length
        df = df[df["length"] >= self.region_length]

        # Get the total lengths of the regions for each chromosome using groupby + sum
        chrom_summary = df.groupby("chr", as_index=False).sum()

        # Create a dictionary from the groupby data for use in calculating the weights
        chromosome_interval_length_dict = pd.Series(chrom_summary.length.values, index=chrom_summary.chr).to_dict()

        # For every interval, calculate the proportion of the total interval lengths as the weight
        df["weights"] = df.apply(lambda row: calculate_weight(row["length"],
                                                              chromosome_interval_length_dict[row["chr"]]), axis=1)

        return df

    def __get_interval_weights(self):
        """
        Get the weights associated with the intervals of interest.
        :return:
        """
        length_sum = 0

        for chromosome in self.chromosomes:
            length_sum += self.chromosome_sizes_dictionary[chromosome]

        weights_list = []

        for chromosome in self.chromosomes:
            weights_list.append(calculate_weight(self.chromosome_sizes_dictionary[chromosome], length_sum))

        self.weights_list = weights_list

    def __get_random_region(self):
        """
        Get a random region from the pool.

        :return: The chromosome string, region start, and region end
        """
        # Randomly select a chromosome from the validation chromosome list with weight proportional to chromosome size
        chrom_name = np.random.choice(self.chromosomes, p=self.weights_list)

        # Create a tmp random regions pool that only has the
        tmp_random_regions_pool = self.preferences_pool[self.preferences_pool["chr"] == chrom_name].sample(n=1, weights="weights")

        # Get the interval from the preferences pool with the longer intervals weighed more
        interval = tmp_random_regions_pool.sample(n=1, weights="weights")

        # Randomly generate a start based on the interval start and stop. Subtract length to prevent out of bounds
        start = random.randint(int(interval["start"]), int(interval["stop"] - (self.region_length + 1)))

        # Find the end point based on the start + region length
        end = start + self.region_length

        return [chrom_name, start, end, "Random", np.random.choice(self.cell_types)]

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


def calculate_weight(length, interval_length_sum):
    """
    Calculate the weights for selecting the random regions from the interval based on the length of the interval
    divided by the total possible lengths for the chromosomes.

    :param length: length of the interval
    :param interval_length_sum: Sum of the lengths of all chromosomes or intervals

    :return: a weight for the interval
    """
    return length / interval_length_sum
