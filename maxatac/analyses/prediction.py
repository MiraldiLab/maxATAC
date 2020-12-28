import logging
import numpy as np
import pandas as pd

from keras.models import load_model

from maxatac.utilities.helpers import build_chrom_sizes_dict

from maxatac.utilities.constants import (
    INPUT_LENGTH,
    BP_RESOLUTION
)

from maxatac.utilities.prepare import (
    get_roi_pool,
    window_prediction_intervals,
    write_df2bigwig,
    get_batch
)

from maxatac.utilities.session import configure_session

def make_predictions(
    signal,
    sequence,
    model,
    predict_roi_df,
    threads, 
    batch_size,
    round
    ):
    configure_session(threads)

    logging.error("Load Model")

    nn_model = load_model(model, compile=False)

    print(nn_model.summary())

    n_batches = int(predict_roi_df.shape[0]/batch_size)

    all_batch_idxs = np.array_split(np.arange(predict_roi_df.shape[0]), n_batches)  
    
    logging.error("Making predictions")

    for idx, batch_idxs in enumerate(all_batch_idxs):
        batch_roi_df = predict_roi_df.loc[batch_idxs, :]

        batch_roi_df.reset_index(drop=True, inplace=True)

        input_batch = get_batch(
                signal=signal,
                sequence=sequence,
                roi_pool=batch_roi_df,
                bp_resolution=BP_RESOLUTION
                )

        pred_output_batch = nn_model.predict(input_batch)

        if idx > 0:
            predictions = np.vstack((predictions, pred_output_batch))

        else:
            predictions = pred_output_batch 

    predictions = np.round(predictions, round)
    logging.error("Parsing results into pandas dataframe")

    predictions_df = pd.DataFrame(data=predictions, index=None, columns=None)

    predictions_df["chr"] = predict_roi_df["chr"]
    predictions_df["start"] = predict_roi_df["start"]
    predictions_df["stop"] = predict_roi_df["stop"]

    return predictions_df 

def run_prediction(args, save_preds=True):
    outfile_name_bigwig = args.output + "/" + args.prefix + ".bw"

    logging.error(
        "Prediction" +
        "\n  Output filename: " + outfile_name_bigwig +
        "\n  Target signal: " + args.signal +
        "\n  Sequence data: " + args.sequence +
        "\n  Models: \n   - " + "\n   - ".join(args.models) +
        "\n  Chromosomes: " + args.chroms +
        "\n  Minimum prediction value to be reported: " + str(args.minimum) +
        "\n  Jobs count: " + str(len(args.chroms) * len(args.models)) +
        "\n  Threads count: " + str(args.threads) +
        "\n  Keep temporary data: " + str(args.keep) +
        "\n  Logging level: " + logging.getLevelName(args.loglevel) +
        "\n  Temp directory: " + args.tmp +
        "\n  Output directory: " + args.output
    )

    roi_df = get_roi_pool(
                        seq_len=INPUT_LENGTH,
                        roi=args.predict_roi,
                        shuffle=False
                        )

    logging.error("Make predictions on ROIs of interest")
    
    pred_results =  make_predictions(
                                        args.signal,
                                        args.sequence,
                                        args.models[0],
                                        roi_df,
                                        args.threads,
                                        args.batch_size,
                                        args.round
                                    )

    logging.error("Convert the predictions to a dataframe that has a bedgraph format")

    df_intervals = window_prediction_intervals(pred_results)

    pred_df = pred_results.drop(['chr', 'start', 'stop'], axis=1)

    df_intervals['score'] = pred_df.to_numpy().flatten()

    df_intervals.columns = ['chr', 'start', 'stop', 'score']
    
    logging.error("Write bedgraph format dataframe to bigwig file")

    write_df2bigwig(output_filename=outfile_name_bigwig, 
                    interval_df=df_intervals, 
                    chromosome_length_dictionary=build_chrom_sizes_dict("hg38"), 
                    chrom=args.chroms)