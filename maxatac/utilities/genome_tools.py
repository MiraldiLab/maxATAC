import pandas as pd
import random
import pybedtools
import numpy as np
import pyBigWig
import py2bit

from maxatac.utilities.system_tools import get_absolute_path


def build_chrom_sizes_dict(chromosome_list,
                           chrom_sizes_filename
                           ):
    """
    Build a dictionary of chromosome sizes.

    :param chromosome_list: (str) path to the chromosome sizes file.
    :param chrom_sizes_filename: (str) a list of chromosomes of interest.
    :return: A dictionary of chromosome sizes filtered by chromosome list.
    """
    chrom_sizes_df = pd.read_csv(chrom_sizes_filename, header=None, names=["chr", "len"], sep="\t")

    chrom_sizes_df = chrom_sizes_df[chrom_sizes_df["chr"].isin(chromosome_list)]

    return pd.Series(chrom_sizes_df.len.values, index=chrom_sizes_df.chr).to_dict()


class DataGenerator(object):
    def __init__(self,
                 meta_table,
                 chroms,
                 blacklist,
                 region_length,
                 average,
                 sequence,
                 batch_size,
                 bp_resolution,
                 input_channels,
                 chrom_pool_size,
                 chrom_sizes,
                 rand_ratio):
        self.chrom_pool_size = chrom_pool_size
        self.input_channels = input_channels
        self.bp_resolution = bp_resolution
        self.batch_size = batch_size
        self.sequence = sequence
        self.average = average
        self.region_length = region_length
        self.blacklist = blacklist
        self.chroms = chroms
        self.meta_table = meta_table
        self.rand_ratio = rand_ratio

        self.n_roi = round(batch_size * (1. - rand_ratio))
        self.n_rand = round(batch_size - self.n_roi)

        self.chrom_sizes_dict = build_chrom_sizes_dict(chroms, chrom_sizes)

        if rand_ratio == 1:
            self.ROI_pool = None
        else:
            self.ROI_pool = self._get_ROIPool()

        if rand_ratio > 0:
            self.RandomRegions_pool = self._get_RandomRegionsPool()
        else:
            self.RandomRegions_pool = None

    def _get_ROIPool(self):
        return ROIPool(meta_path=self.meta_table,
                       chroms=self.chroms,
                       chrom_sizes_dict=self.chrom_sizes_dict,
                       blacklist=self.blacklist,
                       input_length=self.region_length,
                       average=self.average,
                       sequence=self.sequence,
                       batch_size=self.batch_size,
                       bp_resolution=self.bp_resolution,
                       input_channels=self.input_channels
                       )

    def _get_RandomRegionsPool(self):
        return RandomRegionsPool(chrom_sizes_dict=self.chrom_sizes_dict,
                                 chrom_pool_size=self.chrom_pool_size,
                                 region_length=self.region_length,
                                 sequence=self.sequence,
                                 average=self.average,
                                 meta_table=self.meta_table,
                                 input_channels=self.input_channels,
                                 bp_resolution=self.bp_resolution
                                 )

    def BatchGenerator(self):
        while True:
            if 0. < self.rand_ratio < 1.:
                rand_gen = self.RandomRegions_pool.BatchGenerator(n_rand=self.n_rand)
                roi_gen = self.ROI_pool.BatchGenerator(n_roi=self.n_roi)

                roi_input_batch, roi_target_batch = next(roi_gen)
                rand_input_batch, rand_target_batch = next(rand_gen)
                inputs_batch = np.concatenate((roi_input_batch, rand_input_batch), axis=0)
                targets_batch = np.concatenate((roi_target_batch, rand_target_batch), axis=0)

            elif self.rand_ratio == 1.:
                rand_gen = self.RandomRegions_pool.BatchGenerator(n_rand=self.n_rand)

                rand_input_batch, rand_target_batch = next(rand_gen)
                inputs_batch = rand_input_batch
                targets_batch = rand_target_batch

            else:
                roi_gen = self.ROI_pool.BatchGenerator(n_roi=self.n_roi)

                roi_input_batch, roi_target_batch = next(roi_gen)
                inputs_batch = roi_input_batch
                targets_batch = roi_target_batch

            yield inputs_batch, targets_batch


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


def get_input_matrix(rows,
                     cols,
                     batch_size,  # make sure that cols % batch_size == 0
                     signal_stream,
                     average_stream,
                     sequence_stream,
                     bp_order,
                     chrom,
                     start,  # end - start = cols
                     end,
                     reshape=True,

                     ):
    """
    Generate the matrix of values from the signal, sequence, and average data tracks

    :param rows: (int) The number of channels or rows
    :param cols: (int) The number of columns or length
    :param batch_size: (int) The number of examples per batch
    :param signal_stream: (str) ATAC-seq signal
    :param average_stream: (str) Average ATAC-seq signal
    :param sequence_stream: (str) One-hot encoded sequence
    :param bp_order: (list) Order of the bases in matrix
    :param chrom: (str) Chromosome name
    :param start: (str) Chromosome start
    :param end: (str) Chromosome end
    :param reshape: (bool) Whether to transpose the matrix

    :return: A matrix that is rows x columns with the values from each file
    """
    input_matrix = np.zeros((rows, cols))

    for n, bp in enumerate(bp_order):
        input_matrix[n, :] = get_one_hot_encoded(
            sequence_stream.sequence(chrom, start, end),
            bp
        )

    signal_array = np.array(signal_stream.values(chrom, start, end))
    avg_array = np.array(average_stream.values(chrom, start, end))
    input_matrix[4, :] = signal_array
    input_matrix[5, :] = input_matrix[4, :] - avg_array
    input_matrix = input_matrix.T

    if reshape:
        input_matrix = np.reshape(
            input_matrix,
            (batch_size, round(cols / batch_size), rows)
        )

    return input_matrix


class RandomRegionsPool(object):
    """
    This class will generate a pool of random regions

    The RandomRegionsGenerator will generate a random region of interest based
    based on the input reference genome chrom sizes, the desired size of the
    chromosome pool, the input length, and the method used to calculate the
    frequency of chromosome examples.

    Args
    ----
    chrom_sizes_dict (dict):
        A dictionary of chromosome sizes.
    chrom_pool_size (int):
        The size of the chromosome pool to output.
    region_length (int):
        The size of the regions to generate.
    sequence (str):
        The 2bit DNA sequence path.
    average (str):
        The average ATAC-seq sequence path.
    meta_table (str):
        Path to the meta run file.
    input_channels (int):
        The number of input channels.
    bp_resolution (int):
        The resolution of the data to use.
    method (str): optional
        Method for calculating chromosome frequencies.

    Attributes
    ----------
    chrom_sizes_dict (dict):
        A dictionary of chromosome sizes filtered for chromosomes of interest.
    chrom_pool_size (int):
        The size of the chromosome pool to output.
    region_length (int):
        The size of the regions to generate.
    method (str):
        Method for calculating chromosome frequencies.
    chrom_pool (array):
        Chromosome pool that is adjusted by frequency.

    Methods
    -------
    get_region()
        Gets a random region from the available pool.
    BatchGenerator()
        Generates the input and target batches.

    """

    def __init__(
            self,
            chrom_sizes_dict,
            chrom_pool_size,
            region_length,
            sequence,
            average,
            meta_table,
            input_channels,
            bp_resolution,
            method="length"
    ):
        """

        :param chrom_sizes_dict:
        :param chrom_pool_size:
        :param region_length:
        :param sequence:
        :param average:
        :param meta_table:
        :param input_channels:
        :param bp_resolution:
        :param method:
        """
        self.chrom_sizes_dict = chrom_sizes_dict
        self.chrom_pool_size = chrom_pool_size
        self.region_length = region_length
        self.__idx = 0
        self.method = method
        self.sequence = sequence
        self.average = average
        self.MetaPath = meta_table
        self.input_channels = input_channels
        self.bp_resolution = bp_resolution

        self.chrom_pool = self.__get_chrom_frequencies()
        self.MetaDF = self._ImportMeta()
        self.CellTypes = self._get_CellTypes()
        self.TranscriptionFactor = self._get_TranscriptionFactor()

    def _ImportMeta(self):
        return pd.read_csv(self.MetaPath, sep='\t', header=0, index_col=None)

    def _get_CellTypes(self):
        return self.MetaDF["Cell_Line"].unique().tolist()

    def _get_TranscriptionFactor(self):
        """

        :return:
        """
        return self.MetaDF["TF"].unique()[0]

    def __get_chrom_frequencies(self):
        """
        Generate an array of chromosome frequencies

        The frequencies will be used to build the pool of regions that
        we pull random regions from. The frequencies are determined by
        the method attribute. The methods are length or proportion.

        The length method will generate the frequencies of examples in
        the pool based on the length of the chromosomes in total.

        The proportion method will generate a pool that has chromosome
        frequencies equel to the chromosome pools size divided by the
        number of chromosomes.

        Returns
        -------
        labels (list):
            A list of chromosome names the size of the desired pool.

        """

        if self.method == "length":
            sum_lengths = sum(self.chrom_sizes_dict.values())

            frequencies = {
                chrom_name: round(chrom_len / sum_lengths * self.chrom_pool_size)
                for chrom_name, chrom_len in self.chrom_sizes_dict.items()
            }

        else:
            chrom_number = len(self.chrom_sizes_dict.values())

            frequencies = {
                chrom_name: round(self.chrom_pool_size / chrom_number)
                for chrom_name, chrom_len in self.chrom_sizes_dict.items()
            }

        labels = []

        for k, v in frequencies.items():
            labels += [(k, self.chrom_sizes_dict[k])] * v
        random.shuffle(labels)

        return labels

    def get_region(self):
        """
        Get a random region from the chromosome pool.

        Returns
        -------
        chrom_name (str):
            The chromosome string
        start (int):
            The region start
        end (int):
            The region end

        """

        # If the __idx reaches the pool size before enough samples are selected
        # shuffle and set to 0
        if self.__idx == self.chrom_pool_size:
            random.shuffle(self.chrom_pool)
            self.__idx = 0

        chrom_name, chrom_length = self.chrom_pool[self.__idx]
        self.__idx += 1

        start = round(
            random.randint(
                0,
                chrom_length - self.region_length
            )
        )

        end = start + self.region_length

        return chrom_name, start, end

    def BatchGenerator(self, n_rand):
        while True:
            inputs_batch, targets_batch = [], []
            for idx in range(n_rand):
                cell_line = random.choice(self.CellTypes)  # Randomly select a cell line

                chrom_name, start, end = self.get_region()  # returns random region (chrom_name, start, end)

                meta_row = self.MetaDF[self.MetaDF['Cell_Line'] == cell_line]

                meta_row = meta_row.reset_index(drop=True)

                signal = meta_row.loc[0, 'ATAC_Signal_File']

                binding = meta_row.loc[0, 'Binding_File']

                with \
                        load_bigwig(self.average) as average_stream, \
                        load_2bit(self.sequence) as sequence_stream, \
                        load_bigwig(signal) as signal_stream, \
                        load_bigwig(binding) as binding_stream:
                    input_matrix = get_input_matrix(
                        rows=self.input_channels,
                        cols=self.region_length,
                        batch_size=1,  # we will combine into batch later
                        reshape=False,
                        bp_order=["A", "C", "G", "T"],
                        signal_stream=signal_stream,
                        average_stream=average_stream,
                        sequence_stream=sequence_stream,
                        chrom=chrom_name,
                        start=start,
                        end=end
                    )
                    inputs_batch.append(input_matrix)
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    n_bins = int(target_vector.shape[0] / self.bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_sums = np.sum(split_targets, axis=1)
                    bin_vector = np.where(bin_sums > 0.5 * self.bp_resolution, 1.0, 0.0)
                    targets_batch.append(bin_vector)

            yield np.array(inputs_batch), np.array(targets_batch)


class ROIPool(object):
    """
    This class will create an ROI Generator

    Args
    ----
    meta_path (str):
        The path to the meta file.
    chroms (list):
        A list of chromosomes to filter the list by.
    chrom_sizes_dict (dict):
        A dictionary of chromosome sizes that has only the chroms of interest.
    blacklist (str):
        The path to the blacklisted regions for removal.
    input_length (int):
        The desired ROI length.
    average (str):
        The path to the average signal bigwig file.
    sequence (str):
        The path to the 2bit sequence file

    Attributes
    ----------

    Yields
    ------

    """

    def __init__(self,
                 meta_path,
                 chroms,
                 chrom_sizes_dict,
                 blacklist,
                 input_length,
                 average,
                 sequence,
                 batch_size,
                 bp_resolution,
                 input_channels,
                 ):
        self.MetaPath = meta_path
        self.chrom_sizes_dict = chrom_sizes_dict
        self.chroms = chroms
        self.blacklist = blacklist
        self.input_length = input_length
        self.average = average
        self.sequence = sequence
        self.batch_size = batch_size
        self.bp_resolution = bp_resolution
        self.input_channels = input_channels

        self.MetaDF = self._ImportMeta()
        self.CellTypes = self._get_CellTypes()
        self.TranscriptionFactor = self._get_TranscriptionFactor()
        self.PeakPaths = self._get_PeakPaths()
        self.ROI_POOL = self._GetROIPool()

        if self.batch_size == "all":
            self.batch_size = self.ROI_POOL.shape()[0]

        else:
            self.batch_size = batch_size

    def _ImportMeta(self):
        return pd.read_csv(self.MetaPath, sep='\t', header=0, index_col=None)

    def _get_CellTypes(self):
        return self.MetaDF["Cell_Line"].unique().tolist()

    def _get_TranscriptionFactor(self):
        return self.MetaDF["TF"].unique()[0]

    def _get_PeakPaths(self):
        return self.MetaDF["ATAC_Peaks"].unique().tolist() + self.MetaDF["CHIP_Peaks"].unique().tolist()

    def _import_bed(self, bed_file, ):
        df = pd.read_csv(bed_file,
                         sep="\t",
                         usecols=[0, 1, 2],
                         header=None,
                         names=["chr", "start", "stop"],
                         low_memory=False)

        df = df[df["chr"].isin(self.chroms)]

        df["length"] = df["stop"] - df["start"]

        df["center"] = np.floor(df["start"] + (df["length"] / 2)).apply(int)

        df["start"] = np.floor(df["center"] - (self.input_length / 2)).apply(int)

        df["stop"] = np.floor(df["center"] + (self.input_length / 2)).apply(int)

        df["END"] = df["chr"].map(self.chrom_sizes_dict)

        df = df[df["stop"].apply(int) < df["END"].apply(int)]

        df = df[df["start"].apply(int) > 0]

        df = df[["chr", "start", "stop"]]

        BED_df_bedtool = pybedtools.BedTool.from_dataframe(df)

        blacklist_bedtool = pybedtools.BedTool(self.blacklist)

        blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

        df = blacklisted_df.to_dataframe()

        df.columns = ["chr", "start", "stop"]

        return df

    def _GetROIPool(self):
        bed_list = []

        for bed_file in self.PeakPaths:
            bed_list.append(self._import_bed(bed_file))

        return pd.concat(bed_list).sample(frac=1)

    def BatchGenerator(self,
                       n_roi):
        while True:
            inputs_batch, targets_batch = [], []

            roi_size = self.ROI_POOL.shape[0]

            curr_batch_idxs = random.sample(range(roi_size), n_roi)

            # Here I will process by row, if performance is bad then process by cell line
            for row_idx in curr_batch_idxs:
                roi_row = self.ROI_POOL.iloc[row_idx, :]

                cell_line = random.sample(self.CellTypes, 1)[0]

                chrom_name = roi_row['chr']

                start = int(roi_row['start'])

                end = int(roi_row['stop'])

                meta_row = self.MetaDF[self.MetaDF['Cell_Line'] == cell_line]

                meta_row = meta_row.reset_index(drop=True)

                signal = meta_row.loc[0, 'ATAC_Signal_File']

                binding = meta_row.loc[0, 'Binding_File']

                with \
                        load_bigwig(self.average) as average_stream, \
                        load_2bit(self.sequence) as sequence_stream, \
                        load_bigwig(signal) as signal_stream, \
                        load_bigwig(binding) as binding_stream:
                    input_matrix = get_input_matrix(
                        rows=self.input_channels,
                        cols=self.input_length,
                        batch_size=1,  # we will combine into batch later
                        reshape=False,
                        bp_order=["A", "C", "G", "T"],
                        signal_stream=signal_stream,
                        average_stream=average_stream,
                        sequence_stream=sequence_stream,
                        chrom=chrom_name,
                        start=start,
                        end=end
                    )
                    inputs_batch.append(input_matrix)
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    n_bins = int(target_vector.shape[0] / self.bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_sums = np.sum(split_targets, axis=1)
                    bin_vector = np.where(bin_sums > 0.5 * self.bp_resolution, 1.0, 0.0)
                    targets_batch.append(bin_vector)

            yield np.array(inputs_batch), np.array(targets_batch)


def get_significant(data, min_threshold):
    selected = np.concatenate(([0], np.greater_equal(data, min_threshold).view(np.int8), [0]))
    breakpoints = np.abs(np.diff(selected))
    ranges = np.where(breakpoints == 1)[0].reshape(-1, 2)  # [[s1,e1],[s2,e2],[s3,e3]]
    expanded_ranges = list(map(lambda a: list(range(a[0], a[1])), ranges))
    mask = sum(expanded_ranges, [])  # to flatten
    starts = mask.copy()  # copy list just in case
    ends = [i + 1 for i in starts]
    return mask, starts, ends


def window_prediction_intervals(df, number_intervals=32):
    # Create BedTool object from the dataframe
    df_css_bed = pybedtools.BedTool.from_dataframe(df[['chr', 'start', 'stop']])

    # Window the intervals into 32 bins
    pred_css_bed = df_css_bed.window_maker(b=df_css_bed, n=number_intervals)

    # Create a dataframe from the BedTool object
    return pred_css_bed.to_dataframe()


def write_df2bigwig(output_filename, interval_df, chromosome_length_dictionary, chrom):
    with dump_bigwig(output_filename) as data_stream:
        header = [(chrom, int(chromosome_length_dictionary[chrom]))]
        data_stream.addHeader(header)

        data_stream.addEntries(
            chroms=interval_df["chr"].tolist(),
            starts=interval_df["start"].tolist(),
            ends=interval_df["stop"].tolist(),
            values=interval_df["score"].tolist()
        )
