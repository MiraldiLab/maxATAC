import numpy as np
import pandas as pd
import sys
import logging
import random
from os import path
from yaml import dump, safe_load
from maxatac.utilities.bigwig import load_bigwig, safe_load_bigwig
from maxatac.utilities.twobit import load_2bit
from maxatac.utilities.constants import (
    INPUT_CHANNELS,
    INPUT_LENGTH,
    BATCH_SIZE,
    VAL_BATCH_SIZE,
    CHR_POOL_SIZE,
    BP_ORDER,
    OUTPUT_FILTERS,
    CONV_BLOCKS,
    DILATION_RATE,
    BP_RESOLUTION,
    INPUT_FILTERS,
    POOL_SIZE,
    FILTERS_SCALING_FACTOR,
    INPUT_KERNEL_SIZE,
    OUTPUT_KERNEL_SIZE,
    INPUT_ACTIVATION,
    OUTPUT_ACTIVATION,
    PADDING,
    ADAM_BETA_1,
    ADAM_BETA_2,
    TRAIN_MONITOR,
    TRAIN_SCALE_SIGNAL
)
from maxatac.utilities.prepare import (
    get_splitted_chromosomes,
    RandomRegionsPool,
    get_input_matrix
)

def get_roi_pool_predict(seq_len=None, roi=None, shuffle=False, tf=None, cl=None):
    roi_df = pd.read_csv(roi, sep="\t", header=0, index_col=None)
    temp = roi_df['Stop'] - roi_df['Start']
    ##############################
    
    #Temporary Workaround. Needs to be deleted later 
    roi_ok = (temp == seq_len)
    temp_df = roi_df[roi_ok==True]
    roi_df = temp_df
    ###############################

    #roi_ok = (temp == seq_len).all()
    #if not roi_ok:
        
        #sys.exit("ROI Length Does Not Match Input Length")
    roi_df['TF'] = tf
    roi_df['Cell_Line'] = cl
    if shuffle:
        roi_df = roi_df.sample(frac=1)
    return roi_df




def get_roi_pool(seq_len=None, roi=None, shuffle=False):
    roi_df = pd.read_csv(roi, sep="\t", header=0, index_col=None)
    temp = roi_df['Stop'] - roi_df['Start']
    ##############################
    #Temporary Workaround. Needs to be deleted later 
    roi_ok = (temp == seq_len)
    temp_df = roi_df[roi_ok==True]
    roi_df = temp_df
    ###############################

    #roi_ok = (temp == seq_len).all()
    #if not roi_ok:
        
        #sys.exit("ROI Length Does Not Match Input Length")
        
    if shuffle:
        roi_df = roi_df.sample(frac=1)
    return roi_df




def get_one_hot_encoded(sequence, target_bp):
    one_hot_encoded = []
    for s in sequence:
        if s.lower() == target_bp.lower():
            one_hot_encoded.append(1)
        else:
            one_hot_encoded.append(0)
    return one_hot_encoded

def get_pc_input_matrix(
        rows,
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


   

def make_pc_pred_batch(
        batch_idxs,
        sequence,
        average,
        meta_table,
        roi_pool,
        bp_resolution=1,
        filters=None
):
    roi_size = roi_pool.shape[0]
    #batch_idx=0
    #n_batches = int(roi_size/BATCH_SIZE)
    #Here I will process by row, if performance is bad then process by cell line
    
    with \
            safe_load_bigwig(filters) as filters_stream, \
            load_bigwig(average) as average_stream, \
            load_2bit(sequence) as sequence_stream:
        
        inputs_batch, targets_batch = [], []
        batch_meta_df = pd.DataFrame()
        batch_gold_vals = []
        for row_idx in batch_idxs:
            roi_row = roi_pool.iloc[row_idx,:]
            cell_line = roi_row['Cell_Line']
            tf = roi_row['TF']
            chrom_name = roi_row['Chr']
            start = int(roi_row['Start'])
            end = int(roi_row['Stop'])
            meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == tf))]
            meta_row = meta_row.reset_index(drop=True)
            meta_row["Start"] = start
            meta_row["Stop"] = end
            meta_row["Chr"] = chrom_name
            try:
                signal = meta_row.loc[0, 'ATAC_Signal_File']
                binding = meta_row.loc[0, 'Binding_File']
            except:
                print(roi_row)
                
                #sys.exit("Here=1. Error while creating input batch")
    
            
            with \
                    load_bigwig(binding) as binding_stream, \
                    load_bigwig(signal) as signal_stream:
                try:
                    input_matrix = get_pc_input_matrix( rows=INPUT_CHANNELS,
                                                        cols=INPUT_LENGTH,
                                                        batch_size=1,                  # we will combine into batch later
                                                        reshape=False,
                                                        bp_order=BP_ORDER,
                                                        signal_stream=signal_stream,
                                                        average_stream=average_stream,
                                                        sequence_stream=sequence_stream,
                                                        chrom=chrom_name,
                                                        start=start,
                                                        end=end
                                                    )
                    inputs_batch.append(input_matrix)
                    batch_meta_df = pd.concat([batch_meta_df, meta_row], axis='index', ignore_index=True)
                    target_vector = np.array(binding_stream.values(chrom_name, start, end)).T
                    target_vector = np.nan_to_num(target_vector, 0.0)
                    n_bins = int(target_vector.shape[0] / bp_resolution)
                    split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                    bin_sums = np.sum(split_targets, axis=1)
                    bin_vector = np.where(bin_sums > 0.5*bp_resolution, 1.0, 0.0)
                    batch_gold_vals.append(bin_vector)
                    
                except:
                    print(roi_row)
                    continue
                    #sys.exit("Error while creating input batch")
        batch_meta_df = batch_meta_df.drop(['ATAC_Signal_File', 'Binding_File'], axis='columns')
        batch_meta_df.reset_index(drop=True)
        return (np.array(inputs_batch), np.array(batch_gold_vals), batch_meta_df)       
            

def create_roi_batch(
    sequence,
    average,
    meta_table,
    roi_pool,
    n_roi,
    train_tf,
    tchroms,
    bp_resolution=1,
    filters=None
    ):
        
        
        while True:
            inputs_batch, targets_batch = [], []
            roi_size = roi_pool.shape[0]
        
            curr_batch_idxs = random.sample(range(roi_size), n_roi)
    
            #Here I will process by row, if performance is bad then process by cell line
            for row_idx in curr_batch_idxs:
                roi_row = roi_pool.iloc[row_idx,:]
                cell_line = roi_row['Cell_Line']
                tf = train_tf
                chrom_name = roi_row['Chr']
                try:
                    assert chrom_name in tchroms, \
                            "Chromosome in roi file not in tchroms list. Exiting"
                except:
                    #print("Skipped {0} because it is not in tchroms".format(chrom_name))
                    continue
                start = int(roi_row['Start'])
                end = int(roi_row['Stop'])
                meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == tf))]
                meta_row = meta_row.reset_index(drop=True)
                try:
                    signal = meta_row.loc[0, 'ATAC_Signal_File']
                    binding = meta_row.loc[0, 'Binding_File']
                except:
                    print("could not read meta_row. row_idx = {0}".format(row_idx))
                    continue
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
                            batch_size=1,                  # we will combine into batch later
                            reshape=False,
                            bp_order=BP_ORDER,
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
                        n_bins = int(target_vector.shape[0] / bp_resolution)
                        split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                        bin_sums = np.sum(split_targets, axis=1)
                        bin_vector = np.where(bin_sums > 0.5*bp_resolution, 1.0, 0.0)
                        targets_batch.append(bin_vector)

                    except:
                        here = 3
                        print(roi_row)
                        continue
        
            yield (np.array(inputs_batch), np.array(targets_batch))

def create_random_batch(
    sequence,
    average,
    meta_table,
    train_cell_lines,
    n_rand,
    train_tf,
    regions_pool,
    bp_resolution=1,
    filters=None
):
    
    while True:
        inputs_batch, targets_batch = [], []
        for idx in range(n_rand):
            cell_line = random.choice(train_cell_lines) #Randomly select a cell line
            chrom_name, seq_start, seq_end = regions_pool.get_region()  # returns random region (chrom_name, start, end) 
            meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == train_tf))] #get meta table row corresponding to selected cell line
            meta_row = meta_row.reset_index(drop=True)
            signal = meta_row.loc[0, 'ATAC_Signal_File']
            binding = meta_row.loc[0, 'Binding_File']
            with \
                safe_load_bigwig(filters) as filters_stream, \
                load_bigwig(average) as average_stream, \
                load_2bit(sequence) as sequence_stream, \
                load_bigwig(signal) as signal_stream, \
                load_bigwig(binding) as binding_stream:
                    try:
                        input_matrix = get_input_matrix(
                            rows=INPUT_CHANNELS,
                            cols=INPUT_LENGTH,
                            batch_size=1,                  # we will combine into batch later
                            reshape=False,
                            bp_order=BP_ORDER,
                            signal_stream=signal_stream,
                            average_stream=average_stream,
                            sequence_stream=sequence_stream,
                            chrom=chrom_name,
                            start=seq_start,
                            end=seq_end,
                            scale_signal=TRAIN_SCALE_SIGNAL,
                            filters_stream=filters_stream
                        )
                        inputs_batch.append(input_matrix)
                        target_vector = np.array(binding_stream.values(chrom_name, seq_start, seq_end)).T
                        target_vector = np.nan_to_num(target_vector, 0.0)
                        n_bins = int(target_vector.shape[0] / bp_resolution)
                        split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                        bin_sums = np.sum(split_targets, axis=1)
                        bin_vector = np.where(bin_sums > 0.5*bp_resolution, 1.0, 0.0)
                        targets_batch.append(bin_vector)
                    except:
                        here = 2
                        continue
        
        yield (np.array(inputs_batch), np.array(targets_batch))    
        
        
    

def pc_train_generator(
        sequence,
        average,
        meta_table,
        roi_pool,
        train_cell_lines,
        rand_ratio,
        train_tf,
        tchroms,
        bp_resolution=1,
        filters=None
):
    
    
    roi_size = roi_pool.shape[0]
    n_roi = round(BATCH_SIZE*(1. - rand_ratio))
    n_rand = round(BATCH_SIZE - n_roi)
    train_random_regions_pool = RandomRegionsPool(
        chroms=tchroms,
        chrom_pool_size=CHR_POOL_SIZE,
        region_length=INPUT_LENGTH,
        preferences=False                          # can be None
    )

    roi_gen = create_roi_batch( sequence,
                                average,
                                meta_table,
                                roi_pool,
                                n_roi,
                                train_tf,
                                tchroms,
                                bp_resolution=bp_resolution,
                                filters=None
                              )
    
    rand_gen = create_random_batch(  sequence,
                                     average,
                                     meta_table,
                                     train_cell_lines,
                                     n_rand,
                                     train_tf,
                                     train_random_regions_pool,
                                     bp_resolution=bp_resolution,
                                     filters=None
                                  )
                                            
    while True:
        
        #roi_batch.shape = (n_samples, 1024, 6)
        if rand_ratio > 0. and rand_ratio < 1.:
            roi_input_batch, roi_target_batch = next(roi_gen)
            rand_input_batch, rand_target_batch = next(rand_gen)
            inputs_batch = np.concatenate((roi_input_batch, rand_input_batch), axis=0)
            targets_batch = np.concatenate((roi_target_batch, rand_target_batch), axis=0)
        
        elif rand_ratio == 1.:
            rand_input_batch, rand_target_batch = next(rand_gen)
            inputs_batch = rand_input_batch
            targets_batch = rand_target_batch
        
        else:
            roi_input_batch, roi_target_batch = next(roi_gen)
            inputs_batch = roi_input_batch
            targets_batch = roi_target_batch
        
        yield (inputs_batch, targets_batch)
 
      
def create_val_generator(
        sequence,
        average,
        meta_table,
        train_cell_lines,
        train_tf,
        all_val_regions,
        bp_resolution=1,
        filters=None
):

    while True:
        
        inputs_batch, targets_batch = [], []
        n_val_batches = round(all_val_regions.shape[0]/VAL_BATCH_SIZE)
        all_batch_idxs = np.array_split(np.arange(all_val_regions.shape[0]), n_val_batches)  
        
        for idx, batch_idxs in enumerate(all_batch_idxs):
            inputs_batch, targets_batch = [], []
            for row_idx in batch_idxs:
                roi_row = all_val_regions.iloc[row_idx,:]
                cell_line = roi_row['Cell_Line']
                chrom_name = roi_row['Chr']
                seq_start = int(roi_row['Start'])
                seq_end = int(roi_row['Stop'])
                meta_row = meta_table[((meta_table['Cell_Line'] == cell_line) & (meta_table['TF'] == train_tf))]
                meta_row = meta_row.reset_index(drop=True)
                try:
                    signal = meta_row.loc[0, 'ATAC_Signal_File']
                    binding = meta_row.loc[0, 'Binding_File']
                except:
                    print(roi_row)
                   
                with \
                    safe_load_bigwig(filters) as filters_stream, \
                    load_bigwig(average) as average_stream, \
                    load_2bit(sequence) as sequence_stream, \
                    load_bigwig(signal) as signal_stream, \
                    load_bigwig(binding) as binding_stream:
                        input_matrix = get_input_matrix(
                            rows=INPUT_CHANNELS,
                            cols=INPUT_LENGTH,
                            batch_size=1,                  # we will combine into batch later
                            reshape=False,
                            bp_order=BP_ORDER,
                            signal_stream=signal_stream,
                            average_stream=average_stream,
                            sequence_stream=sequence_stream,
                            chrom=chrom_name,
                            start=seq_start,
                            end=seq_end,
                            scale_signal=None,
                            filters_stream=filters_stream
                        )
                        inputs_batch.append(input_matrix)
                        target_vector = np.array(binding_stream.values(chrom_name, seq_start, seq_end)).T
                        target_vector = np.nan_to_num(target_vector, 0.0)
                        n_bins = int(target_vector.shape[0] / bp_resolution)
                        split_targets = np.array(np.split(target_vector, n_bins, axis=0))
                        bin_sums = np.sum(split_targets, axis=1)
                        bin_vector = np.where(bin_sums > 0.5*bp_resolution, 1.0, 0.0)
                        targets_batch.append(bin_vector)
                        
            yield (np.array(inputs_batch), np.array(targets_batch))    
    
        
    
        
    
    
        
    
