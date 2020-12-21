import logging
import shutil
import numpy as np
import pandas as pd
from os import path
import os
from math import ceil
from uuid import uuid4
from multiprocessing import Pool

from keras.models import load_model
from maxatac.utilities.helpers import get_dir, get_rootname, load_bigwig, dump_bigwig, load_2bit, Mute
from maxatac.utilities.constants import (
    INPUT_CHANNELS,
    INPUT_LENGTH,
    BATCH_SIZE,
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
    get_input_matrix,
    get_significant
)
from maxatac.utilities.prepare import (
    get_roi_pool,
    get_roi_pool_predict,
    make_pc_pred_batch
)
from maxatac.utilities.session import configure_session

def do_pc_prediction(
    job_id,
    average,
    sequence,
    model,
    predict_roi_df,
    meta_table,
    tmp,
    threads
):
    
    
    
    #logging.error(
    #"Start job [" + str(job_id) + "]" +
    #    "\n  Input signal: " + signal +
    #    "\n  Average signal: " + average +
    #    "\n  Sequence data: " + sequence +
    #    "\n  Model: " + model +
    #    "\n  Chromosome: " + \
    #         chrom[0] + ":" + \
    #         str(region[0]) + "-" + \
    #         str(region[1]) + " (" + \
    #         str(chrom[1]) + ")" +
    #    "\n  Results location: " + results_location
    #)
    
    configure_session(threads)

    nn_model = load_model(model, compile=False)
    print(nn_model.summary())
    n_batches = int(predict_roi_df.shape[0]/BATCH_SIZE)
    print(n_batches)
    all_batch_idxs = np.array_split(np.arange(predict_roi_df.shape[0]), n_batches)  
    print(len(all_batch_idxs))
    pred_df = pd.DataFrame()
    gold_df = pd.DataFrame()
    for idx, batch_idxs in enumerate(all_batch_idxs):
        
        input_batch, batch_gold_vals, batch_meta_df = make_pc_pred_batch(
                                        batch_idxs,
                                        sequence,
                                        average,
                                        meta_table,
                                        predict_roi_df,
                                        bp_resolution=BP_RESOLUTION
                                        )

        pred_output_batch = nn_model.predict(input_batch)
        temp_pred_df = pd.DataFrame(data=pred_output_batch, index=None, columns=None)
        temp_pred_df.reset_index(drop=True)
        batch_pred_df = pd.concat([batch_meta_df, temp_pred_df], axis='columns', ignore_index=False)
        
        temp_gold_df = pd.DataFrame(data=batch_gold_vals, index=None, columns=None)
        temp_gold_df.reset_index(drop=True)
        batch_gold_df = pd.concat([batch_meta_df, temp_gold_df], axis='columns', ignore_index=False)

        pred_df = pd.concat([pred_df, batch_pred_df], axis='index', ignore_index=True)
        gold_df = pd.concat([gold_df, batch_gold_df], axis='index', ignore_index=True)
        print("pred_df.shape={0}, gold_df.shape={1}, batch_idx={2}/{3}".format(pred_df.shape, gold_df.shape, idx, n_batches))
        
    return (pred_df, gold_df)     

    

def export_prediction_results(prediction_results, args):
    logging.error("Export prediction results" + \
         "\n  Min reported prediction: " + str(args.minimum))
    
    prediction_per_chrom = {}  # refactore to dict for easy access
    for chrom_name, location in prediction_results:
        prediction_per_chrom.setdefault(chrom_name, []).append(location)
    
    results_location = path.join(
        get_dir(args.output),
        get_rootname(args.signal) + "_prediction.bigwig"
    )

    with dump_bigwig(results_location) as data_stream:
        header = [
            (n, args.chroms[n]["length"]) for n in sorted(prediction_per_chrom.keys())
        ]
        data_stream.addHeader(header)

        for chrom_name, chrom_length in header:
            region = args.chroms[chrom_name]["region"]
            logging.error("  Processing: " + \
                chrom_name + ":" + str(region[0]) + "-" + str(region[1])
            )

            locations = prediction_per_chrom[chrom_name]
            chrom_prediction = np.zeros(chrom_length)
            for l in locations:
                chrom_prediction[region[0]:region[1]] += np.nan_to_num(
                    np.load(l)[region[0]:region[1]]
                )
            chrom_prediction = chrom_prediction / float(len(locations))

            mask, starts, ends = get_significant(chrom_prediction, args.minimum)
            
            if len(mask) == 0:
                logging.error("  Skipping: no significant prediction found")
                continue

            data_stream.addEntries(
                chroms = [chrom_name] * len(mask),
                starts = starts,
                ends = ends,
                values = chrom_prediction[mask].tolist()
            )

    logging.error("  Results are saved to: " + results_location)
    
    if not args.keep:
        logging.error("Removing temporary data:\n  " + args.tmp)
        shutil.rmtree(args.tmp, ignore_errors=False)


'''
def get_scattered_smart_params(args):
    scattered_params = []
    job_id = 1
    
    roi_df = get_roi_pool_predict(
                                seq_len=INPUT_LENGTH,
                                roi=args.predict_roi,
                                shuffle=False
                                )
            
    for model in args.models:
        for chrom_name, chrom_data in args.chroms.items():
            chrom_roi_df = roi_df.loc[chrom_name, :]
            scattered_params.append(
                (
                    job_id,
                    args.signal,
                    args.average,
                    args.sequence,
                    model,
                    chrom_roi_df,
                    args.tmp,
                    args.threads
                )
            )
            job_id += 1
    return scattered_params
'''
def run_prediction(args, save_preds=True):
    '''
    logging.error(
        "Prediction" +
        "\n  Average signal: " + args.average +
        "\n  Sequence data: " + args.sequence +
        "\n  Models: \n   - " + "\n   - ".join(args.models) +
        "\n  Minimum prediction value to be reported: " + str(args.minimum) +
        "\n  Jobs count: " + str(len(args.chroms) * len(args.models)) +
        "\n  Threads count: " + str(args.threads) +
        "\n  Keep temporary data: " + str(args.keep) +
        "\n  Logging level: " + logging.getLevelName(args.loglevel) +
        "\n  Temp directory: " + args.tmp +
        "\n  Output directory: " + args.output
    )
    '''
    job_id = 1
    average = args.average
    sequence = args.sequence
    model = args.models[0]
    tf = args.train_tf
    #Hard coded for now may need to be changed if there are more than one cell lines used in prediction
    cl = args.test_cell_lines[0]

    roi_df = get_roi_pool_predict(
                        seq_len=INPUT_LENGTH,
                        roi=args.predict_roi,
                        shuffle=False,
                        tf=tf,
                        cl=cl
                        )
    meta_table = pd.read_csv(args.meta_file, sep='\t', header=0, index_col=None)
    tmp = args.tmp
    threads = args.threads
    
    pred_results_df, gold_df =  do_pc_prediction(
                                        job_id,
                                        average,
                                        sequence,
                                        model,
                                        roi_df,
                                        meta_table,
                                        tmp,
                                        threads
                                        )
    
    '''
    results_filename = get_rootname(model) + \
        "_" + get_rootname(signal) + \
        "_" + get_rootname(average) + \
        "_" + str(uuid4()) + ".npy"  # TODO: think again why do we need uuid here
    
    results_location = path.join(get_dir(tmp), results_filename)
    '''
    if save_preds:
        #out = pred_results_df['chr'].unique()
        #outname='_'.join(map(str, out))
        #outname = outname+"_pc_predictions.bed"
        
        out_path = os.path.join(args.output, "pcpc_predictions.bed")
        pred_results_df.to_csv(out_path, sep='\t')
        
        out_path = os.path.join(args.output, "pcpc_gold.bed")
        gold_df.to_csv(out_path, sep='\t')
    

    '''
    with Pool(args.threads) as p:
        prediction_results = p.starmap(
            run_model_prediction,
            get_scattered_params(args)
        )
    export_prediction_results(prediction_results, args)
    '''