from deeplift.dinuc_shuffle import dinuc_shuffle  # function to do a dinucleotide shuffle
import numpy as np
from multiprocessing import Pool
from functools import partial

from maxatac.utilities.constants import INPUT_CHANNELS, INPUT_LENGTH, BP_ORDER
from maxatac.utilities.genome_tools import safe_load_bigwig, load_bigwig, load_2bit
from maxatac.utilities.training_tools import get_pc_input_matrix


def dinuc_shuffle_DNA_only_several_times(list_containing_input_modes_for_an_example,
                                         seed=1234):
    assert len(list_containing_input_modes_for_an_example) == 1
    onehot_seq = list_containing_input_modes_for_an_example[0][:, :4]  # DNA only: length x 4
    ATAC_signals = list_containing_input_modes_for_an_example[0][:, 4:]  # ATAC-seq signal only: length x 2
    rng = np.random.RandomState(seed)
    to_return = np.array([np.concatenate((dinuc_shuffle(onehot_seq, rng=rng),
                                          ATAC_signals), axis=1) for _ in range(10)])
    return [to_return]


# shap explainer's combine_mult_and_diffref function
def combine_DNA_only_mult_and_diffref(mult, orig_inp, bg_data):
    assert len(orig_inp) == 1
    projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")
    assert len(orig_inp[0].shape) == 2
    for i in range(4):
        hypothetical_input = np.zeros_like(orig_inp[0]).astype("float")  # length x 6  all set to 0
        hypothetical_input[:, i] = 1.0  # change only DNA position
        hypothetical_input[:, -2:] = orig_inp[0][:, -2:]  # copy over ATAC signal
        hypothetical_difference_from_reference = (hypothetical_input[None, :, :] - bg_data[0])
        hypothetical_contribs = hypothetical_difference_from_reference * mult[0]
        projected_hypothetical_contribs[:, :, i] = np.sum(hypothetical_contribs, axis=-1)
    return [np.mean(projected_hypothetical_contribs, axis=0)]


def output_meme_pwm(pwm, pattern_name):
    with open(pattern_name + '.meme', 'w') as f:
        f.writelines('MEME version 4\n')
        f.writelines('\n')
        f.writelines('ALPHABET= ACGT\n')
        f.writelines('\n')
        f.writelines('strands: + -\n')
        f.writelines('\n')
        f.writelines('Background letter frequencies (from uniform background):\n')
        f.writelines('A 0.25000 C 0.25000 G 0.25000 T 0.25000 \n')
        f.writelines('\n')

        l = np.shape(pwm)[0]

        f.writelines('MOTIF {} {}\n'.format(pattern_name, pattern_name))
        f.writelines('\n')
        f.writelines('letter-probability matrix: alength= 4 w= {} nsites= 1 E= 0\n'.format(l))
        for i in range(0, l):
            _sum = np.sum([pwm[i, 0], pwm[i, 1], pwm[i, 2], pwm[i, 3]])
            f.writelines('  {}	  {}	  {}	  {}	\n'.format(float(pwm[i, 0]) / _sum, float(pwm[i, 1]) / _sum,
                                                                    float(pwm[i, 2]) / _sum, float(pwm[i, 3]) / _sum))
        f.writelines('\n')


def generating_interpret_data(sequence,
                              average,
                              meta_table,
                              roi_pool,
                              train_cell_lines,
                              rand_ratio,
                              train_tf,
                              tchroms,
                              bp_resolution=1,
                              filters=None,
                              workers=8
                              ):
    _mp = Pool(workers)
    _data = np.array(_mp.map(partial(process_map,
                                     sequence=sequence,
                                     average=average,
                                     meta_table=meta_table,
                                     roi_pool=roi_pool,
                                     train_tf=train_tf,
                                     bp_resolution=bp_resolution,
                                     filters=filters
                                     ), roi_pool.index[:])
                     )
    _mp.close()
    _mp.join()
    return (_data[:, 0], _data[:, 1])


def process_map(row_idx,
                sequence,
                average,
                meta_table,
                roi_pool,
                train_tf,
                bp_resolution,
                filters=None):
    # print('processing: ', row_idx)
    roi_row = roi_pool.iloc[row_idx, :]
    cell_line = roi_row['Cell_Line']
    tf = train_tf
    chrom_name = roi_row['Chr']

    start = int(roi_row['Start'])
    end = int(roi_row['Stop'])
    meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == tf))]
    meta_row = meta_row.reset_index(drop=True)
    try:
        signal = meta_row.loc[0, 'ATAC_Signal_File']
        binding = meta_row.loc[0, 'Binding_File']
    except:
        print("could not read meta_row. row_idx = {0}".format(row_idx))
    with \
            safe_load_bigwig(filters) as filters_stream, \
            load_bigwig(average) as average_stream, \
            load_2bit(sequence) as sequence_stream, \
            load_bigwig(signal) as signal_stream, \
            load_bigwig(binding) as binding_stream:
        try:
            input_matrix = get_pc_input_matrix(
                rows=INPUT_CHANNELS,
                cols=INPUT_LENGTH,
                batch_size=1,  # we will combine into batch later
                reshape=False,
                bp_order=BP_ORDER,
                signal_stream=signal_stream,
                average_stream=average_stream,
                sequence_stream=sequence_stream,
                chrom=chrom_name,
                start=start,
                end=end
            )
            # inputs_batch.append(input_matrix)
            target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
            target_vector = np.nan_to_num(target_vector, 0.0)
            n_bins = int(target_vector.shape[0] / bp_resolution)
            split_targets = np.array(np.split(target_vector, n_bins, axis=0))
            bin_sums = np.sum(split_targets, axis=1)
            bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)
            # targets_batch.append(bin_vector)

        except:
            print(roi_row)

    return [input_matrix, bin_vector]
