import os
import pandas as pd
import pybedtools
import numpy as np

from maxatac.utilities.system_tools import get_dir


class GenomicRegions(object):
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
        and validation regions of interest. These will be output in the form of TSV formatted file similar to a BED file.

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
        # Select Training Cell lines
        self.meta_dataframe = self.meta_dataframe[self.meta_dataframe["Train_Test_Label"] == 'Train']

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
            bed_list.append(self.__import_bed(bed_file,
                                              ROI_type_tag=roi_type_tag,
                                              ROI_cell_tag=roi_cell_tag))

        return pd.concat(bed_list)

    def write_data(self, prefix="ROI_pool", output_dir="./ROI", set_tag="training"):
        """
        Write the ROI dataframe to a tsv and a bed for for ATAC, CHIP, and combined ROIs

        :param set_tag: Tag for training or validation
        :param prefix: Prefix for filenames to use
        :param output_dir: Directory to output the bed and tsv files

        :return: Write BED and TSV versions of the ROI data
        """
        output_directory = get_dir(output_dir)

        combined_BED_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_ROI.bed.gz")

        stats_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_ROI_stats")
        total_regions_stats_filename = os.path.join(output_directory,
                                                    prefix + "_" + set_tag + "_ROI_totalregions_stats")

        self.combined_pool.to_csv(combined_BED_filename, sep="\t", index=False, header=False)

        group_ms = self.combined_pool.groupby(["Chr", "Cell_Line", "ROI_Type"], as_index=False).size()
        len_ms = self.combined_pool.shape[0]
        group_ms.to_csv(stats_filename, sep="\t", index=False)

        file = open(total_regions_stats_filename, "a")
        file.write('Total number of regions found for ' + set_tag + ' are: {0}\n'.format(len_ms))
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

    def __import_bed(self,
                     bed_file,
                     ROI_type_tag,
                     ROI_cell_tag):
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

        # Find the length of the regions
        df["length"] = df["Stop"] - df["Start"]

        # Find the center of each peak.
        # We might want to use bedtools to window the regions of interest around the peak.
        df["center"] = np.floor(df["Start"] + (df["length"] / 2)).apply(int)

        # The start of the interval will be the center minus 1/2 the desired region length.
        df["Start"] = np.floor(df["center"] - (self.region_length / 2)).apply(int)

        # the end of the interval will be the center plus 1/2 the desired region length
        df["Stop"] = np.floor(df["center"] + (self.region_length / 2)).apply(int)

        # The chromosome end is defined as the chromosome length
        df["END"] = df["Chr"].map(self.chromosome_sizes_dictionary)

        # Make sure the stop is less than the end
        df = df[df["Stop"].apply(int) < df["END"].apply(int)]

        # Make sure the start is greater than the chromosome start of 0
        df = df[df["Start"].apply(int) > 0]

        # Select for the first three columns to clean up
        df = df[["Chr", "Start", "Stop"]]

        # Import the dataframe as a pybedtools object so we can remove the blacklist
        BED_df_bedtool = pybedtools.BedTool.from_dataframe(df)

        # Import the blacklist as a pybedtools object
        blacklist_bedtool = pybedtools.BedTool(self.blacklist)

        # Find the intervals that do not intersect blacklisted regions.
        blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

        # Convert the pybedtools object to a pandas dataframe.
        df = blacklisted_df.to_dataframe()

        # Rename the columns
        df.columns = ["Chr", "Start", "Stop"]

        df["ROI_Type"] = ROI_type_tag

        df["Cell_Line"] = ROI_cell_tag

        return df
