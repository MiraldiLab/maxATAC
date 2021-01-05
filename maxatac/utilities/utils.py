import pandas as pd
import sys
import logging
import random
import pybedtools
import numpy as np

from os import path

from maxatac.utilities.dcnn import get_callbacks, get_dilated_cnn

from maxatac.utilities.helpers import (get_absolute_path,
                                       get_dir,
                                       replace_extension,
                                       remove_tags
                                       )

from maxatac.utilities.session import configure_session

from maxatac.utilities.plot import (
    export_model_loss,
    export_model_accuracy,
    export_model_dice,
    export_model_structure
)

from maxatac.utilities.prepare import RandomRegionsGenerator, create_roi_batch


def TrainingDataGenerator(
        sequence,
        average,
        meta_table,
        roi_pool,
        rand_ratio,
        chroms,
        batch_size,
        chrom_pool_size,
        bp_resolution,
        region_length
    ):
    """
    Generate data for model training and validation

    Args
    ----
        sequence (str):
            Input 2-bit DNA sequence
        average (str):
            Input average ATAC-seq signal
        meta_table (obj):
            Input meta table object
        roi_pool (list):
            A pool of regions of interest
        rand_ratio (float):
            Proportion of training examples randomly generated
        chroms (list):
            A list of chromosomes of interest
        batch_size (int):
            The number of examples to use per batch
        bp_resolution (int):
            The resolution of the predictions

    Yields
    ------

    """
    n_roi = round(batch_size*(1. - rand_ratio))
    
    n_rand = round(batch_size - n_roi)
    
    roi_gen = create_roi_batch( sequence,
                                average,
                                meta_table,
                                roi_pool,
                                n_roi,
                                chroms,
                                bp_resolution=bp_resolution,
                                filters=None
                              )

    train_random_regions_pool = RandomRegionsGenerator(
        chrom_sizes_dict=build_chrom_sizes_dict(chroms),
        chrom_pool_size=chrom_pool_size,
        region_length=region_length
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
 