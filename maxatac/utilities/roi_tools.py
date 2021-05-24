import os
import pandas as pd
import pybedtools
import numpy as np
import pybedtools

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
                 region_length,
                 window_sequence
                 ):
        """
        When the object is initialized it will import all of the peaks in the meta files and parse them into training
        and validation regions of interest. These will be output in the form of TSV formatted file similar to a BED file.

        :param meta_path: Path to the meta file
        :param chromosomes: List of chromosomes to use
        :param chromosome_sizes_dictionary: A dictionary of chromosome sizes
        :param blacklist: The blacklist file of BED regions to exclude
        :param region_length: Length of the input regions
        :param window_sequence: Windowed hg38 sequence w1024 sliding 256 bp
        """
        self.meta_path = meta_path
        self.chromosome_sizes_dictionary = chromosome_sizes_dictionary
        self.chromosomes = chromosomes
        self.blacklist = blacklist
        self.region_length = region_length
        self.window_sequence = window_sequence

        # Import meta txt as dataframe
        self.meta_dataframe = pd.read_csv(self.meta_path, sep='\t', header=0, index_col=None)

        # Get a dictionary of {Cell Types: Peak Paths}

        #just import the one file
        
        self.combined_pool = self.__import_bed(self.meta_dataframe.All_ChIP_ATAC_Peaks.values[0])
        
        # You must generate the ROI pool before you can get the final shape
        # self.atac_chip_roi_pool = self.__get_roi_pool(self.atac_chip_dictionary, "ATAC_CHIP", )
        '''
        self.atac_roi_pool = self.__get_roi_pool(self.atac_dictionary, "ATAC", )
        self.chip_roi_pool = self.__get_roi_pool(self.chip_dictionary, "CHIP")
        '''
        
            
        #self.combined_pool = pd.concat([self.atac_roi_pool, self.chip_roi_pool])

        #self.atac_roi_size = self.atac_roi_pool.shape[0]
        #self.chip_roi_size = self.chip_roi_pool.shape[0]

    '''def __get_roi_pool(self, dictionary, roi_type_tag): 
        """
        Build a pool of regions of interest from BED files.

        :param dictionary: A dictionary of Cell Types and their associated BED files
        :param roi_type_tag: Tag used to name the type of ROI being generated. IE Chip or ATAC

        :return: A dataframe of BED regions that are formatted for maxATAC training.
        """
        bed_list = []
        
        self.__import_bed(bed_file, ROI_type_tag=roi_type_tag, ROI_cell_tag=roi_cell_tag)
        
        for roi_cell_tag, bed_file in dictionary.items():
            bed_list.append(self.__import_bed(bed_file,
                                              ROI_type_tag=roi_type_tag,
                                              ROI_cell_tag=roi_cell_tag))

        return pd.concat(bed_list)'''

    def write_data(self, prefix="ROI_pool", output_dir="./ROI", set_tag="training"):
        """
        Write the ROI dataframe to a tsv and a bed for for ATAC, CHIP, and combined ROIs

        :param set_tag: Tag for training or validation
        :param prefix: Prefix for filenames to use
        :param output_dir: Directory to output the bed and tsv files

        :return: Write BED and TSV versions of the ROI data
        """
        output_directory = get_dir(output_dir)

        atac_BED_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_ATAC_ROI.bed.gz")
        chip_BED_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_CHIP_ROI.bed.gz")
        combined_BED_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_ROI.bed.gz")

        atac_TSV_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_ATAC_ROI.tsv.gz")
        chip_TSV_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_CHIP_ROI.tsv.gz")
        combined_TSV_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_ROI.tsv.gz")

        stats_filename = os.path.join(output_directory, prefix + "_" + set_tag + "_ROI_stats.tsv.gz")
        total_regions_stats_filename = os.path.join(output_directory,
                                                    prefix + "_" + set_tag + "_ROI_totalregions_stats.tsv.gz")

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
        file.write('Total number of regions found for ' + set_tag + ' are: {0}\n'.format(len_ms))
        file.close()

    '''def get_regions_list(self,
                         n_roi):
        """
        Generate a batch of regions of interest from the input ChIP-seq and ATAC-seq peaks

        :param n_roi: Number of regions to generate per batch

        :return: A batch of training examples centered on regions of interest
        """
        random_roi_pool = self.combined_pool.sample(n=n_roi, replace=True, random_state=1)

        return random_roi_pool.to_numpy().tolist()'''

    def __import_bed(self,
                     bed_file
                     ):
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
                         header=None,
                         names=["Chr", "Start", "Stop", "Chr_overlap", "Start_overlap", "Stop_overlap", "ROI_Type", "Cell_Line"],
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
        df = df[["Chr", "Start", "Stop", "ROI_Type", "Cell_Line"]]

        # Import the dataframe as a pybedtools object so we can remove the blacklist
        BED_df_bedtool = pybedtools.BedTool.from_dataframe(df)

        # Import the blacklist as a pybedtools object
        blacklist_bedtool = pybedtools.BedTool(self.blacklist)

        # Find the intervals that do not intersect blacklisted regions.
        blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

        # Convert the pybedtools object to a pandas dataframe.
        df = blacklisted_df.to_dataframe()
        

        # Rename the columns
        df.columns = ["Chr", "Start", "Stop", "ROI_Type", "Cell_Line"]
        
        '''
        df["ROI_Type"] = ROI_type_tag

        df["Cell_Line"] = ROI_cell_tag
        '''
        
        ###Here we separte out the types of ROIs by Sets
        #Set1: True Positives -- Cell Type Regions that have a ChIP-seq Peak
        #Set2: True Negatives -- For the TP regions, if that region is not a ChIP peak in a different CT included in this bin
        #Set3: ATAC Only: -- Removing the first two Sets, all regions with an ATAC peak and no ChIP peak
        #Set4: Random: -- Regions that are not a TP
        
        #Find the total Cell Lines that are available for Training (excluding the test CL)
        list_CL= df['Cell_Line'].unique().tolist()
        
        #Set1
        Set1_df=[]
        Set1_df_peakcounts=[]
        for CT in list_CL:
            DF=df.copy()
            #Find all CHIP Peaks
            DF=DF[DF['ROI_Type']=='CHIP']
            #Isolate by CT
            DF=DF[DF['Cell_Line'] == CT]
            #Append results for each CT
            Set1_df.append(DF)
            
            #Gather stats
            peakcount=len(DF)
            pc=pd.DataFrame({'CT': CT, 'peakcount': peakcount}, index=[0])
            Set1_df_peakcounts.append(pc)
        Set1_df = pd.concat(Set1_df)
        Set1_df = Set1_df.reset_index(drop=True)
        
        Set1_df_peakcounts = pd.concat(Set1_df_peakcounts)
        Set1_df_peakcounts = Set1_df_peakcounts.reset_index(drop=True)

        
        #Set2
        Set2_df=[]
        Set2_df_peakcounts=[]
        for CT in list_CL:
            DF=df.copy()
            #Find all CHIP Peaks
            DF=DF[DF['ROI_Type']=='CHIP']
            #Remove  that CT for TN
            DF=DF[~DF['Cell_Line'].str.contains(CT)]
            Set2_df.append(DF)
            
            #Gather stats
            peakcount=len(DF)
            pc=pd.DataFrame({'not_CT': CT, 'peakcount': peakcount}, index=[0])
            Set2_df_peakcounts.append(pc)
        Set2_df = pd.concat(Set2_df)
        Set2_df = Set2_df.reset_index(drop=True)
        
        Set2_df_peakcounts = pd.concat(Set2_df_peakcounts)
        Set2_df_peakcounts = Set2_df_peakcounts.reset_index(drop=True)

        #Set3
        #Read in Entire genome w1024 s256 and convert to a bedtool object
        windowed_gen = pd.read_csv(self.window_sequence, sep = '\t', header = None, names=["Chr", "Start", "Stop"])
        windowed_gen_bedtool = pybedtools.BedTool.from_dataframe(windowed_gen)
        
        #Take all regions in Set1 and convert into a bedtool object
        Set1_df_format = Set1_df[['Chr', 'Start', 'Stop']]
        Set1_df_bedtool = pybedtools.BedTool.from_dataframe(Set1_df_format)
        
        #Take all regions in Set2 and convert into a bedtool object
        Set2_df_format = Set2_df[['Chr', 'Start', 'Stop']]
        Set2_df_bedtool = pybedtools.BedTool.from_dataframe(Set2_df_format)
        
        #Remove all bins from the genome that are found in Set1 and Set2
        windowed_gen_bedtool_rmSet1 = windowed_gen_bedtool.intersect(Set1_df_bedtool, v=True)
        windowed_gen_bedtool_rmSet1_rmSet2 = windowed_gen_bedtool_rmSet1.intersect(Set2_df_bedtool, v=True)
        
        #Now windowed_gen_bedtool_rmSet1_rmSet2 is the entire genome without TP from Set1 and TN from Set2
        #To Find ATAC only bins we will intersect with all ATAC peaks
        
        #Find all ATAC peaks and convert to a bedtool
        df_atac=df[df['ROI_Type']=='ATAC']
        df_atac_bedtool = pybedtools.BedTool.from_dataframe(df_atac)
        
        #Intersect and return overlapping bins 'wa' option. Return to a df
        Set3_bedtool = windowed_gen_bedtool_rmSet1_rmSet2.intersect(df_atac_bedtool, wa = True)
        Set3_df = Set3_bedtool.to_dataframe()
        cols = ['Chr', 'Start', 'Stop']
        Set3_df.columns = cols
        
        #Gather Stats on this Set
        Set3_peakcounts = len(Set3_df)
        Set3_df_peakcounts= pd.DataFrame({'ATAC_only': 'All regions with an ATAC peak and no ChIP peak', 'peakcount': Set3_peakcounts}, index=[0])
        #Set4
        
        #Set1_df_format = Set1_df[['Chr', 'Start', 'Stop']]
        #Set1_df_bedtool = pybedtools.BedTool.from_dataframe(Set1_df_format)
        
        #Call on windowed_gen_bedtool all hg38 without regions from Set1 as Set4 random regions
        Set4_bedtool = windowed_gen_bedtool_rmSet1
        Set4_df = Set4_bedtool.to_dataframe()
        Set4_df.columns = cols
        
        #Gather Stats on this Set
        Set4_peakcounts = len(Set4_df)
        Set4_df_peakcounts=pd.DataFrame({'Random': 'Regions that are not a TP', 'peakcount': Set4_peakcounts}, index=[0])

        self.regions1 = Set1_df
        self.regions2 = Set2_df
        self.regions3 = Set3_df
        self.regions4 = Set4_df
        
        self.regions1_counts = Set1_df_peakcounts
        self.regions2_counts = Set2_df_peakcounts
        self.regions3_counts = Set3_df_peakcounts
        self.regions4_counts = Set4_df_peakcounts
