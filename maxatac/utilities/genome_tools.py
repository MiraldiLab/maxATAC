import pandas as pd
import numpy as np
import pyBigWig
import py2bit
import random

import pybedtools

from maxatac.utilities.system_tools import get_absolute_path


def build_chrom_sizes_dict(chromosome_list, chrom_sizes_filename):
    """
    Build a dictionary of chromosome sizes filtered for chromosomes in the input chromosome_list. 
    
    The dictionary takes the form of: 
    
        {
         "chr1": 248956422,
         "chr2": 242193529
        }

    :param chromosome_list: list of chromosome to filter dictionary by
    :param chrom_sizes_filename: path to the chromosome sizes file

    :return: A dictionary of chromosome sizes filtered by chromosome list.
    """
    # Import the data as pandas dataframe
    chrom_sizes_df = pd.read_csv(chrom_sizes_filename, header=None, names=["chr", "len"], sep="\t")

    # Filter the dataframe for the chromosomes of interest
    chrom_sizes_df = chrom_sizes_df[chrom_sizes_df["chr"].isin(chromosome_list)]

    return pd.Series(chrom_sizes_df.len.values, index=chrom_sizes_df.chr).to_dict()


def dump_bigwig(location):
    """
    Write a bigwig file to the location

    :param location: the path to desired file location

    :return: an opened bigwig for writing
    """
    return pyBigWig.open(get_absolute_path(location), "w")


def get_one_hot_encoded(sequence, target_bp):
    """
    Convert a 2bit DNA sequence to a one-hot encoded sequence.

    :param sequence: path to the 2bit DNA sequence
    :param target_bp: resolution of the bp sequence

    :return: one-hot encoded DNA sequence
    """
    one_hot_encoded = []
    for s in sequence:
        if s.lower() == target_bp.lower():
            one_hot_encoded.append(1)
        else:
            one_hot_encoded.append(0)
    return one_hot_encoded


def load_2bit(location):
    """
    Load a 2bit file.

    :param location: path to the 2bit DNA sequence

    :return: opened 2bit file
    """
    return py2bit.open(get_absolute_path(location))


def load_bigwig(location):
    """
    Load a bigwig file

    :param location: path to the bigwig file

    :return: opened bigwig file
    """
    return pyBigWig.open(get_absolute_path(location))


def get_bigwig_values(bigwig_path, chrom_name, chrom_end, chrom_start=0):
    """
    Get the values for a genomic region of interest from a bigwig file.

    :param bigwig_path: Path to the bigwig file
    :param chrom_name: Chromosome name
    :param chrom_end: chromosome end
    :param chrom_start: chromosome start

    :return: Bigwig values from the region given
    """
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.nan_to_num(input_bw.values(chrom_name, chrom_start, chrom_end, numpy=True))


def get_bigwig_stats(bigwig_path, chrom_name, chrom_end, bin_count, agg_function="max"):
    """
    Get the values for a genomic region of interest from a bigwig file.

    :param bin_count:
    :param agg_function:
    :param bigwig_path: Path to the bigwig file
    :param chrom_name: Chromosome name
    :param chrom_end: chromosome end

    :return: Bigwig values from the region given
    """
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.nan_to_num(np.array(input_bw.stats(chrom_name,
                                                     0,
                                                     chrom_end,
                                                     type=agg_function,
                                                     nBins=bin_count,
                                                     exact=True),
                                      dtype=float  # need it to have NaN instead of None
                                      ))


def get_input_matrix(rows,
                     cols,
                     signal_stream,
                     sequence_stream,
                     chromosome,
                     start,  # end - start = cols
                     end,
                     scale_signal
                     ):
    """
    Generate the matrix of values from the signal, sequence, and average data tracks

    :param rows: (int) The number of channels or rows
    :param cols: (int) The number of columns or length
    :param signal_stream: (str) ATAC-seq signal
    :param sequence_stream: (str) One-hot encoded sequence
    :param chromosome: (str) Chromosome name
    :param start: (str) Chromosome start
    :param end: (str) Chromosome end
    :param scale_signal: (tuple) Randomly scale input signal by these values

    :return: A matrix that is rows x columns with the values from each file
    """
    input_matrix = np.zeros((rows, cols))

    for n, bp in enumerate(["A", "C", "G", "T"]):
        input_matrix[n, :] = get_one_hot_encoded(
            sequence_stream.sequence(chromosome, start, end),
            bp
        )

    signal_array = np.array(signal_stream.values(chromosome, start, end))
    input_matrix[4, :] = signal_array

    if scale_signal is not None:
        scaling_factor = random.random() * (scale_signal[1] - scale_signal[0]) + scale_signal[0]
        input_matrix[4, :] = input_matrix[4, :] * scaling_factor

    return input_matrix.T


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


def get_synced_chroms(chroms, ignore_regions=None):
    """
    This function will generate a nested dictionary of chromosome sizes and the regions available for training.

        {
            "chr2": {"length": 243199373, "region": [0, 243199373]},
            "chr3": {"length": 198022430, "region": [0, 198022430]}
        }

    If ignore_regions is True, set regions to the whole chromosome length
    Returns something like this

    """
    chroms_and_regions = {}
    for chrom in chroms:
        chrom_name, *region = chrom.replace(",", "").split(":")  # region is either [] or ["start-end", ...]
        chroms_and_regions[chrom_name] = None
        if not ignore_regions:
            try:
                chroms_and_regions[chrom_name] = [int(i) for i in region[0].split("-")]
            except (IndexError, ValueError):
                pass

    loaded_chroms = set()

    synced_chroms = {}
    for chrom_name, chrom_length in loaded_chroms:
        if chrom_name not in chroms_and_regions: continue
        region = chroms_and_regions[chrom_name]
        if not region or \
                region[0] < 0 or \
                region[1] <= 0 or \
                region[0] >= region[1] or \
                region[1] > chrom_length:
            region = [0, chrom_length]
        synced_chroms[chrom_name] = {
            "length": chrom_length,
            "region": region
        }
    return synced_chroms


class EmptyStream():
    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        pass


def safe_load_bigwig(location):
    try:
        return pyBigWig.open(get_absolute_path(location))
    except (RuntimeError, TypeError):
        return EmptyStream()


def chromosome_blacklist_mask(blacklist, chromosome, chromosome_length, nBins=False, agg_method="max"):
    """
    Import the chromosome signal from a blacklist bigwig file and convert to a numpy array to use to generate the array
    to use to exclude regions. If a number of bins are provided, then the function will use the stats method from 
    pyBigWig to bin the data. 

    :return: blacklist_mask: A np.array the has True for regions that are NOT in the blacklist.
    """
    with load_bigwig(blacklist) as blacklist_bigwig_stream:
        if nBins:
            return np.array(blacklist_bigwig_stream.stats(chromosome,
                                                          0,
                                                          chromosome_length,
                                                          type=agg_method,
                                                          nBins=nBins
                                                          ),
                            dtype=float  # need it to have NaN instead of None
                            ) != 1  # Convert to boolean array, select areas that are not 1

        else:
            return blacklist_bigwig_stream.values(chromosome,
                                                  0,
                                                  chromosome_length,
                                                  numpy=True) != 1  # Convert to boolean array, select areas that are not 1


class GenomicBins(object):
    def __init__(self,
                 chromosome_sizes_file: str,
                 chromosomes: list,
                 blacklist_path: str,
                 window_size=1024,
                 step_size=256):
        """
        Create the genomic bins that will be used for training.

        :param chromosome_sizes_file: Path to the chromosome sizes file
        :param chromosomes: Chromosomes to limit the analysis to
        :param blacklist_path: Path to the blacklist regions file
        :param window_size: Size of the bins desired in base pairs
        :param step_size: Step size to slide bins
        """
        self.chromosome_sizes_file = chromosome_sizes_file
        self.chromosomes = chromosomes
        self.blacklist_path = blacklist_path
        self.window_size = window_size
        self.step_size = step_size

        # Create a bedtools object of blacklisted regions
        self.blacklist_regions = pybedtools.BedTool(self.blacklist_path)

        self.__make_windows__()

    def __make_windows__(self):

        genomic_bins = pybedtools.BedTool().makewindows(g=self.chromosome_sizes_file,
                                                        w=self.window_size,
                                                        s=self.step_size)

        # Remove bins that are overlapping blacklisted regions
        blacklist_genomic_bins = genomic_bins.intersect(self.blacklist_regions, v=True).to_dataframe()

        # Remove undesirable chromosomes
        blacklist_genomic_bins = blacklist_genomic_bins[blacklist_genomic_bins["chrom"].isin(self.chromosomes)]

        self.bins_DF = blacklist_genomic_bins

        self.bins_bedtool = pybedtools.BedTool.from_dataframe(blacklist_genomic_bins)


class RegionsOfInterest(object):
    def __init__(self,
                 meta_path: str,
                 chromosomes: list,
                 blacklist: str
                 ):
        """
        This class will generate a pool of examples based on regions of interest defined by ATAC-seq and ChIP-seq peaks.

        When the object is initialized it will import all of the peaks in the meta files and parse them into training
        and validation regions of interest.

        :param meta_path: Path to the meta file
        :param chromosomes: List of chromosomes to use
        :param blacklist: The blacklist file of BED regions to exclude
        """
        self.meta_path = meta_path
        self.chromosomes = chromosomes
        self.blacklist = blacklist

        # Import meta txt as dataframe
        self.meta_dataframe = pd.read_csv(self.meta_path, sep='\t', header=0, index_col=None)

        # Get a dictionary of {Cell Types: Peak Paths}
        self.atac_dictionary = pd.Series(self.meta_dataframe.ATAC_Peaks.values,
                                         index=self.meta_dataframe.Cell_Line).to_dict()

        self.chip_dictionary = pd.Series(self.meta_dataframe.CHIP_Peaks.values,
                                         index=self.meta_dataframe.Cell_Line).to_dict()

        # Import the blacklist as a pybedtools object
        self.blacklist_bedtool = pybedtools.BedTool(self.blacklist)

        # You must generate the ROI pool before you can get the final shape
        self.atac_roi_pool = self.__get_roi_pool(self.atac_dictionary, "ATAC")
        self.chip_roi_pool = self.__get_roi_pool(self.chip_dictionary, "CHIP")

        self.combined_pool = pd.concat([self.atac_roi_pool, self.chip_roi_pool])

        self.combined_pool_bedtool = pybedtools.BedTool.from_dataframe(self.combined_pool)

        self.atac_roi_size = self.atac_roi_pool.shape[0]
        self.chip_roi_size = self.chip_roi_pool.shape[0]

    def __get_roi_pool(self, dictionary, roi_type_tag):
        """
        Build a pool of regions of interest from BED files.

        :param dictionary: A dictionary of Cell Types and their associated BED file paths
        :param roi_type_tag: Tag used to name the type of ROI being generated. IE Chip or ATAC

        :return: A dataframe of BED regions that are formatted for maxATAC training.
        """
        bed_list = []

        for roi_cell_tag, bed_file in dictionary.items():
            bed_list.append(self.__import_bed__(bed_file,
                                                ROI_type_tag=roi_type_tag,
                                                ROI_cell_tag=roi_cell_tag))

        return pd.concat(bed_list)

    def __import_bed__(self,
                       bed_file: str,
                       ROI_type_tag: str,
                       ROI_cell_tag: str):
        """
        Import a BED file and format the regions to be compatible with our maxATAC models

        :param bed_file: Input BED file to format
        :param ROI_type_tag: Tag to use in the description column
        :param ROI_cell_tag: Tag to use in the description column

        :return: A dataframe of BED regions compatible with our model
        """
        # Import dataframe
        df = pd.read_csv(bed_file,
                         sep="\t",
                         usecols=[0, 1, 2],
                         header=None,
                         names=["Chr", "Start", "Stop"],
                         low_memory=False)

        # Make sure the chromosomes in the ROI file frame are in the target chromosome list
        df = df[df["Chr"].isin(self.chromosomes)]

        # Select for the first three columns to clean up
        df = df[["Chr", "Start", "Stop"]]

        # Import the dataframe as a pybedtools object so we can remove the blacklist
        BED_df_bedtool = pybedtools.BedTool.from_dataframe(df)

        # Find the intervals that do not intersect blacklisted regions.
        blacklisted_df = BED_df_bedtool.intersect(self.blacklist_bedtool, v=True)

        # Convert the pybedtools object to a pandas dataframe.
        df = blacklisted_df.to_dataframe()

        # Rename the columns
        df.columns = ["Chr", "Start", "Stop"]

        df["ROI_Type"] = ROI_type_tag

        df["Cell_Line"] = ROI_cell_tag

        return df
