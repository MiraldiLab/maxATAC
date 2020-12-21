import random
import numpy as np

from maxatac.utilities.bigwig import load_bigwig


class RandomRegionsPool:

    def __init__(
        self,
        chroms,            # in a form of {"chr1": {"length": 249250621, "region": [0, 249250621]}}, "region" is ignored
        chrom_pool_size,
        region_length,
        preferences=None   # bigBed file with ranges to limit random regions selection
    ):

        
        self.chroms = chroms
        self.chrom_pool_size = chrom_pool_size
        self.region_length = region_length
        self.preferences = preferences

        #self.preference_pool = self.__get_preference_pool()  # should be run before self.__get_chrom_pool()
        self.preference_pool = False
        
        self.chrom_pool = self.__get_chrom_pool()

        self.__idx = 0


    def get_region(self):
        
        
        if self.__idx == self.chrom_pool_size:
            random.shuffle(self.chrom_pool)
            self.__idx = 0

        chrom_name, chrom_length = self.chrom_pool[self.__idx]
        self.__idx += 1

        if self.preference_pool:
            preference = random.sample(self.preference_pool[chrom_name], 1)[0]
            start = round(
                random.randint(
                    preference[0],
                    preference[1] - self.region_length
                )
            )
        else:
            start = round(
                random.randint(
                    0,
                    chrom_length - self.region_length
                )
            )
        end = start + self.region_length

        return (chrom_name, start, end)


    def __get_preference_pool(self):
        preference_pool = {}
        if self.preferences is not None:
            with load_bigwig(self.preferences) as input_stream:
                for chrom_name, chrom_data in self.chroms.items():
                    for entry in input_stream.entries(
                        chrom_name,
                        0,
                        chrom_data["length"],
                        withString=False
                    ):
                        if entry[1] - entry[0] < self.region_length:
                            continue
                        preference_pool.setdefault(
                            chrom_name, []
                        ).append(list(entry[0:2]))
        return preference_pool


    def __get_chrom_pool(self):
        """
        TODO: rewrite to produce exactly the same number of items
        as chrom_pool_size regardless of length(chroms) and
        chrom_pool_size
        """
        
        chroms = {
            chrom_name: chrom_data
            for chrom_name, chrom_data in self.chroms.items()
            #if not self.preference_pool or (chrom_name in self.preference_pool)
        }

        sum_lengths = sum(map(lambda v: v["length"], chroms.values()))
        frequencies = {
            chrom_name: round(
                chrom_data["length"] / sum_lengths * self.chrom_pool_size
            )
            for chrom_name, chrom_data in chroms.items()
        }
        labels = []
        for k, v in frequencies.items():
            labels += [(k, chroms[k]["length"])] * v
        random.shuffle(labels)
        
        return labels


def get_significant(data, min_threshold):
    selected = np.concatenate(([0], np.greater_equal(data, min_threshold).view(np.int8), [0]))
    breakpoints = np.abs(np.diff(selected))
    ranges = np.where(breakpoints == 1)[0].reshape(-1, 2)  # [[s1,e1],[s2,e2],[s3,e3]]
    expanded_ranges = list(map(lambda a : list(range(a[0], a[1])), ranges))
    mask = sum(expanded_ranges, [])  # to flatten
    starts = mask.copy()  # copy list just in case
    ends = [i + 1 for i in starts]
    return mask, starts, ends


def get_one_hot_encoded(sequence, target_bp):
    one_hot_encoded = []
    for s in sequence:
        if s.lower() == target_bp.lower():
            one_hot_encoded.append(1)
        else:
            one_hot_encoded.append(0)
    return one_hot_encoded


def get_splitted_chromosomes(
    chroms,
    tchroms,
    vchroms,
    proportion
):
    """
    Doesn't take regions into account.
    May produce not correct results if inputs are not received from
    get_synced_chroms with ignore_regions=True
    """
    free_chrom_set = set(chroms) - set(tchroms) - set(vchroms)

    n = round(len(free_chrom_set) * proportion)

    # need sorted list for random.sample to reproduce in tests
    tchrom_set = set(random.sample(sorted(free_chrom_set), n))
    vchrom_set = free_chrom_set.difference(tchrom_set).union(set(vchroms))
    tchrom_set = tchrom_set.union(set(tchroms))

    extended_tchrom = {
        chrom_name: {
            "length": chroms[chrom_name]["length"],
            "region": chroms[chrom_name]["region"]
        } for chrom_name in tchrom_set}

    extended_vchrom = {
        chrom_name: {
            "length": chroms[chrom_name]["length"],
            "region": chroms[chrom_name]["region"]
        } for chrom_name in vchrom_set}

    return extended_tchrom, extended_vchrom


def get_input_matrix(
    rows,
    cols,
    batch_size,          # make sure that cols % batch_size == 0
    signal_stream,
    average_stream,
    sequence_stream,
    bp_order,
    chrom,
    start,               # end - start = cols
    end,
    reshape=True,
    scale_signal=None,   # (min, max) ranges to scale signal
    filters_stream=None  # defines regions that should be set to 0
):
 
    input_matrix = np.zeros((rows, cols))
    for n, bp in enumerate(bp_order):
        input_matrix[n, :] = get_one_hot_encoded(
            sequence_stream.sequence(chrom, start, end),
            bp
        )
    signal_array = np.array(signal_stream.values(chrom, start, end))
    avg_array = np.array(average_stream.values(chrom, start, end))

    if filters_stream is not None:
        exclude_mask = np.array(filters_stream.values(chrom, start, end)) <= 0
        signal_array[exclude_mask] = 0
        avg_array[exclude_mask] = 0

    input_matrix[4, :] = signal_array
    input_matrix[5, :] = input_matrix[4, :] - avg_array

    if scale_signal is not None:
        scaling_factor = random.random() * (scale_signal[1] - scale_signal[0]) + \
            scale_signal[0]
        input_matrix[4, :] = input_matrix[4, :] * scaling_factor

    input_matrix = input_matrix.T

    if reshape:
        input_matrix = np.reshape(
            input_matrix,
            (batch_size, round(cols/batch_size), rows)
        )

    return input_matrix
