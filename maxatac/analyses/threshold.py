import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool, Manager
import time
from maxatac.utilities.genome_tools import build_chrom_sizes_dict, get_bigwig_stats
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.threshold_tools import import_blacklist_mask, import_GoldStandard_array, calculate_AUC_per_rank, compute_calibration_curve, bin_curve_by_metric, extend_bins_to_full_grid, bin_median_table_by_f1, merge_binned_metrics, build_cross_cell_type_threshold_table
from maxatac.utilities.plot import plot_threshold_calibration_stats
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import pybedtools
import logging


def extract_pred_gs_bw(bigwig_file, training_data_dict, chrom_name, chrom_length, bin_count):
    start = time.time()
    bw_name = bigwig_file.split("/")[-1]
    chrom_vals = get_bigwig_stats(bigwig_file, chrom_name, chrom_length, bin_count)
    print(training_data_dict[bigwig_file])
    
    
    predictions = np.empty(1, dtype=np.float64)
    gold_standard = np.empty(1, dtype=np.float64)
    goldstandard_array = import_GoldStandard_array(training_data_dict[bigwig_file], chrom_name, chrom_length, bin_count)
    
    tot_gs_bins = len(np.argwhere(goldstandard_array == True))

    predictions = np.concatenate([predictions, chrom_vals])
    gold_standard = np.concatenate([gold_standard, goldstandard_array])
    
    end = time.time()
    print('total time (s)= ' + str(end-start), "____________", bw_name)
    
    return predictions, gold_standard, tot_gs_bins

def run_thresholding(args):
    """
    :param args:
    :return:
    """
    # Make the output directory
    output_dir = get_dir(args.output_dir)

    chromosome_sizes_dictionary = build_chrom_sizes_dict(args.chromosomes, args.chrom_sizes)

    meta_DF = pd.read_table(args.meta_file)

    training_data_dict = pd.Series(meta_DF["Binding_File"].values,index=meta_DF["Prediction"]).to_dict()

    # Loop through the chromosomes and average the values across files
    OUT=[]
    for chrom_name, chrom_length in chromosome_sizes_dictionary.items():
        bin_count = int(int(chrom_length) / int(args.bin_size))  # need to floor the number
        
        blacklist_mask = import_blacklist_mask(args.blacklist_bw, chrom_name, chrom_length, bin_count)

        lst_of_bws=list(training_data_dict.keys())
        
        pool = Pool(int(multiprocessing.cpu_count())) 
        output = pool.starmap(
            extract_pred_gs_bw,
            [(bigwig, training_data_dict, chrom_name, chrom_length, bin_count) for bigwig in lst_of_bws]
                            )
        OUT.append(output)


    # Stack each cell type's Prediction/GoldStandard as a pair of columns
    DF=pd.DataFrame([])
    total_gs_bins = []
    for i in range(len(OUT[0])):
        df = pd.DataFrame([])

        df['Prediction'] = OUT[0][i][0][:bin_count].tolist()
        df['GoldStandard'] = OUT[0][i][1][:bin_count].tolist()

        gs_bins = OUT[0][i][2]
        DF = pd.concat([DF, df], axis=1, ignore_index=True)
        total_gs_bins.append(gs_bins)

    num_cell_types = len(OUT[0])

    # Create a bedtools object that is a windowed genome
    BED_df_bedtool = pybedtools.BedTool().window_maker(g=args.chrom_sizes, w=args.bin_size)

    # Create a blacklist object form the blacklist bed
    # TODO: I do not think we need to get the blacklist bed location like this since it is a part of the args now.
    blacklist_bed_location = ".".join([args.blacklist_bw.split(".")[0],'bed'])
    blacklist_bedtool = pybedtools.BedTool(blacklist_bed_location)

    # Remove the blacklisted regions from the windowed genome object
    blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

    # Create a dataframe from the BedTools object
    df = blacklisted_df.to_dataframe()

    # Rename the columns
    df.columns = ["chr", "start", "stop"]

    # Find the number of non-blacklisted bins in chr of interest (log2FC denominator)
    rand_bins = df.query('chr == @args.chromosomes').shape[0]

    logging.info("Building per-cell-type calibration curves")

    raw_cell_type_curves = []
    for i in range(num_cell_types):
        ct_curve = compute_calibration_curve(
            DF[2 * i + 1][blacklist_mask], DF[2 * i][blacklist_mask], total_gs_bins[i], rand_bins
        )
        raw_cell_type_curves.append(ct_curve)

    logging.info("Building cross-cell-type threshold table (per-metric binning + median across cell types)")

    # Bin each cell type's curve by Precision/Recall/F1 (0.01 steps), then take the
    # median across cell types per bin. Only thresholds are aggregated, never raw signal.
    cross_celltype_table = build_cross_cell_type_threshold_table(raw_cell_type_curves)
    cross_celltype_filename = os.path.join(output_dir, args.prefix + "_cross_celltype.tsv")
    cross_celltype_table.to_csv(cross_celltype_filename, sep="\t", header=True, index=False)

    logging.info("Plotting the validation statistics v. threshold values")

    # Plot each cell type's own binned+extended curve rather than resampling it at the
    # median thresholds, so one cell type's gaps cannot distort another's line.
    cell_type_curves = []
    for i, ct_curve in enumerate(raw_cell_type_curves):
        random_precision_i = total_gs_bins[i] / rand_bins
        precision_curve = extend_bins_to_full_grid(bin_curve_by_metric(ct_curve, 'Precision'))
        recall_curve = extend_bins_to_full_grid(bin_curve_by_metric(ct_curve, 'Recall'))

        # Save this cell type's own binned threshold table (same schema as cross_celltype.tsv)
        sample_table = merge_binned_metrics([precision_curve, recall_curve, bin_median_table_by_f1([recall_curve])])
        sample_name = os.path.splitext(os.path.basename(lst_of_bws[i]))[0]
        sample_table.to_csv(os.path.join(output_dir, sample_name + ".tsv"), sep="\t", header=True, index=False)

        # F1 is not monotonic in Threshold, so re-binning it by value zigzags; the
        # Recall-binned curve already carries a Threshold-monotonic F1 column, reuse it.
        curves_by_metric = {'Precision': precision_curve, 'Recall': recall_curve, 'F1': recall_curve.copy()}
        for curve in curves_by_metric.values():
            curve['log2FC'] = np.log2(curve['Precision'] / random_precision_i)
        cell_type_curves.append({'name': os.path.basename(lst_of_bws[i]), 'curves': curves_by_metric})

    # log2FC is needed for the plot only; not written to cross_celltype.tsv
    random_precision = np.mean(total_gs_bins) / rand_bins
    median_curve_for_plot = cross_celltype_table.copy()
    median_curve_for_plot['log2FC'] = np.log2(median_curve_for_plot['Precision'] / random_precision)

    plot_threshold_calibration_stats(median_curve_for_plot, cell_type_curves, cross_celltype_filename, args.prefix)


