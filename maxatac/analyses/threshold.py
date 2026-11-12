import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool
import time
from maxatac.utilities.genome_tools import build_chrom_sizes_dict, get_bigwig_stats
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.threshold_tools import import_blacklist_mask, import_GoldStandard_array, calculate_AUC_per_rank, compute_calibration_curve, bin_curve_by_metric, extend_bins_to_full_grid, bin_median_table_by_f1, merge_binned_metrics, build_cross_cell_type_threshold_table, import_peak_bed_bins, gold_standard_run_bins, peak_max_predictions, peak_recall_at_thresholds
from maxatac.utilities.plot import plot_threshold_calibration_stats
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import pybedtools
import logging


def extract_pred_gs_bw(bigwig_file, training_data_dict, chrom_name, chrom_length, bin_count,
                       peaks_bed=None, blacklist_mask=None):
    """
    Bin one cell type's prediction bigwig and its binary gold standard on one chromosome, and
    reduce the prediction to one max value per unique ChIP-seq peak (for Peak_Recall).

    Peaks come from peaks_bed (unique BED intervals) when given, else from runs of consecutive
    gold-standard bins. Blacklisted bins are ignored; fully blacklisted peaks are dropped.

    :return: (predictions, gold_standard, tot_gs_bins, peak_max, peak_source) with predictions
             and gold_standard both of length bin_count and aligned bin-for-bin.
    """
    start = time.time()
    bw_name = os.path.basename(bigwig_file)

    predictions = get_bigwig_stats(bigwig_file, chrom_name, chrom_length, bin_count)
    gold_standard = import_GoldStandard_array(training_data_dict[bigwig_file], chrom_name, chrom_length, bin_count)

    tot_gs_bins = int(np.count_nonzero(gold_standard))

    if peaks_bed:
        peak_idx, bin_idx = import_peak_bed_bins(peaks_bed, chrom_name, chrom_length, bin_count)
        peak_source = os.path.basename(peaks_bed)
    else:
        peak_idx, bin_idx = gold_standard_run_bins(gold_standard)
        peak_source = "gold_standard_runs"
    peak_max = peak_max_predictions(predictions, peak_idx, bin_idx, keep_mask=blacklist_mask)

    logging.info(f"Binned {bw_name} vs {os.path.basename(training_data_dict[bigwig_file])} "
                 f"on {chrom_name} in {time.time() - start:.1f} s; "
                 f"{len(peak_max)} unique peaks from {peak_source}")

    return predictions, gold_standard, tot_gs_bins, peak_max, peak_source

def run_thresholding(args):
    """
    :param args:
    :return:
    """
    # Calibration is done on exactly one held-out chromosome: the per-cell-type arrays,
    # the blacklist mask and the random-precision denominator all refer to that chromosome.
    if len(args.chromosomes) != 1:
        raise ValueError("maxatac threshold calibrates on exactly one chromosome; "
                         f"got --chromosomes {' '.join(args.chromosomes)}")

    # Make the output directory
    output_dir = get_dir(args.output_dir)

    chromosome_sizes_dictionary = build_chrom_sizes_dict(args.chromosomes, args.chrom_sizes)
    (chrom_name, chrom_length), = chromosome_sizes_dictionary.items()

    meta_DF = pd.read_table(args.meta_file)

    training_data_dict = pd.Series(meta_DF["Binding_File"].values,index=meta_DF["Prediction"]).to_dict()

    # Optional ChIP_peaks column (BED) defines the unique peaks for Peak_Recall; a missing
    # column or empty cell falls back to runs of consecutive gold-standard bins.
    if "ChIP_peaks" in meta_DF.columns:
        peaks_dict = {pred: (bed if isinstance(bed, str) and bed.strip() else None)
                      for pred, bed in zip(meta_DF["Prediction"], meta_DF["ChIP_peaks"])}
    else:
        peaks_dict = {}

    bin_count = int(int(chrom_length) / int(args.bin_size))  # need to floor the number

    blacklist_mask = import_blacklist_mask(args.blacklist_bw, chrom_name, chrom_length, bin_count)

    lst_of_bws = list(training_data_dict.keys())

    # Bin every cell type's prediction and gold standard in parallel
    with Pool(int(multiprocessing.cpu_count())) as pool:
        output = pool.starmap(
            extract_pred_gs_bw,
            [(bigwig, training_data_dict, chrom_name, chrom_length, bin_count,
              peaks_dict.get(bigwig), blacklist_mask) for bigwig in lst_of_bws]
        )

    # Stack each cell type's Prediction/GoldStandard as a pair of columns
    DF = pd.DataFrame([])
    total_gs_bins = []
    peak_maxes = []
    peak_sources = []
    for predictions, gold_standard, gs_bins, peak_max, peak_source in output:
        df = pd.DataFrame({'Prediction': predictions, 'GoldStandard': gold_standard})
        DF = pd.concat([DF, df], axis=1, ignore_index=True)
        total_gs_bins.append(gs_bins)
        peak_maxes.append(peak_max)
        peak_sources.append(peak_source)

    num_cell_types = len(output)

    # Create a bedtools object that is a windowed genome
    BED_df_bedtool = pybedtools.BedTool().window_maker(g=args.chrom_sizes, w=args.bin_size)

    # Create a blacklist object form the blacklist bed: --blacklist_bed if given, otherwise the .bed next to --blacklist_bw.
    # splitext only strips the final extension, so dots elsewhere in the path are kept.
    blacklist_bed_location = args.blacklist_bed or os.path.splitext(args.blacklist_bw)[0] + ".bed"
    blacklist_bedtool = pybedtools.BedTool(blacklist_bed_location)

    # Remove the blacklisted regions from the windowed genome object
    blacklisted_df = BED_df_bedtool.intersect(blacklist_bedtool, v=True)

    # Create a dataframe from the BedTools object
    df = blacklisted_df.to_dataframe()

    # Rename the columns
    df.columns = ["chr", "start", "stop"]

    # Find the number of non-blacklisted bins in the chromosome of interest (log2FC denominator)
    rand_bins = df.query('chr == @chrom_name').shape[0]

    logging.info("Building per-cell-type calibration curves")

    raw_cell_type_curves = []
    for i in range(num_cell_types):
        ct_curve = compute_calibration_curve(
            DF[2 * i + 1][blacklist_mask], DF[2 * i][blacklist_mask], total_gs_bins[i], rand_bins,
            peak_max=peak_maxes[i]
        )
        raw_cell_type_curves.append(ct_curve)

    logging.info("Building cross-cell-type threshold table (per-metric binning + median across cell types)")

    # Bin each cell type's curve by Precision/Recall/F1 (0.01 steps), then take the
    # median across cell types per bin. Only thresholds are aggregated, never raw signal.
    cross_celltype_table = build_cross_cell_type_threshold_table(raw_cell_type_curves)
    cross_celltype_filename = os.path.join(output_dir, args.prefix + "_cross_celltype.tsv")
    cross_celltype_table.to_csv(cross_celltype_filename, sep="\t", header=True, index=False)

    # Per-cell-type % of unique ChIP-seq peaks recovered at its own max-F1 threshold and at
    # the cross-cell-type max-F1 threshold
    cross_f1 = cross_celltype_table[cross_celltype_table['Metric'] == 'F1']
    cross_best_threshold = cross_f1.loc[cross_f1['F1'].idxmax(), 'Threshold']
    peak_recovery_rows = []
    for i, ct_curve in enumerate(raw_cell_type_curves):
        own_best = ct_curve.loc[ct_curve['F1'].idxmax()]
        peak_recovery_rows.append({
            'Prediction': os.path.basename(lst_of_bws[i]),
            'Peak_Source': peak_sources[i],
            'N_Peaks': len(peak_maxes[i]),
            'MaxF1_Threshold': own_best['Threshold'],
            'MaxF1': own_best['F1'],
            'Peak_Recall_at_MaxF1': own_best['Peak_Recall'],
            'CrossCellType_MaxF1_Threshold': cross_best_threshold,
            'Peak_Recall_at_CrossCellType_MaxF1': peak_recall_at_thresholds(peak_maxes[i], [cross_best_threshold])[0],
        })
    pd.DataFrame(peak_recovery_rows).to_csv(os.path.join(output_dir, args.prefix + "_peak_recovery.tsv"),
                                            sep="\t", header=True, index=False)

    logging.info("Plotting the validation statistics v. threshold values")

    # Plot each cell type's own binned+extended curve rather than resampling it at the
    # median thresholds, so one cell type's gaps cannot distort another's line.
    cell_type_curves = []
    for i, ct_curve in enumerate(raw_cell_type_curves):
        precision_curve = extend_bins_to_full_grid(bin_curve_by_metric(ct_curve, 'Precision'))
        recall_curve = extend_bins_to_full_grid(bin_curve_by_metric(ct_curve, 'Recall'))
        peak_recall_curve = extend_bins_to_full_grid(bin_curve_by_metric(ct_curve, 'Peak_Recall'))

        # Save this cell type's own binned threshold table (same schema as cross_celltype.tsv)
        sample_table = merge_binned_metrics([precision_curve, recall_curve, bin_median_table_by_f1([recall_curve]),
                                             peak_recall_curve])
        sample_name = os.path.splitext(os.path.basename(lst_of_bws[i]))[0]
        sample_table.to_csv(os.path.join(output_dir, sample_name + ".tsv"), sep="\t", header=True, index=False)

        # F1 is not monotonic in Threshold, so re-binning it by value zigzags; the
        # Recall-binned curve already carries a Threshold-monotonic F1 column, reuse it.
        curves_by_metric = {'Precision': precision_curve, 'Recall': recall_curve, 'F1': recall_curve.copy(),
                            'Peak_Recall': peak_recall_curve}
        cell_type_curves.append({'name': os.path.basename(lst_of_bws[i]), 'curves': curves_by_metric})

    plot_threshold_calibration_stats(cross_celltype_table, cell_type_curves, cross_celltype_filename, args.prefix,
                                     chrom_name=chrom_name)


