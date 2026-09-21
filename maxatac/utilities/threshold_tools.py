import numpy as np
import pandas as pd
import pyBigWig
from sklearn import metrics
from sklearn.metrics import precision_recall_curve


def f1_from_precision_recall(precision, recall):
    """
    F1 = 2PR / (P + R), defined as 0 where P = R = 0 (e.g. a cell type with no reachable bins)
    instead of NaN, so a single degenerate row cannot propagate through the binning, the
    cross-cell-type medians, or the max-F1 lookups.
    """
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)
    denominator = precision + recall

    return np.divide(2 * precision * recall, denominator,
                     out=np.zeros_like(denominator), where=denominator > 0)


def import_blacklist_mask(bigwig_path, chromosome, chromosome_length, bin_count):
    """
        Import the chromosome signal from a blacklist bigwig file and convert to a numpy array to use to mask out
        the regions to exclude in the AUPR analysis
        :return: blacklist_mask: A np.array the has True for regions that should be excluded from analysis
        """
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.array(input_bw.stats(chromosome,
                                       0,
                                       chromosome_length,
                                       type="max",
                                       nBins=bin_count
                                       ),
                        dtype=float  # need it to have NaN instead of None
                        ) != 1  # Convert to boolean array, select areas that are not 1


def import_GoldStandard_array(bigwig_path, chromosome, chromosome_length, bin_count):
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.nan_to_num(np.array(input_bw.stats(chromosome,
                                                     0,
                                                     chromosome_length,
                                                     type="max",
                                                     nBins=bin_count,
                                                     exact=True
                                                     ),
                                      dtype=float  # need it to have NaN instead of None
                                      )
                             ) > 0  # to convert to boolean array


def calculate_AUC_per_rank(PR_CURVE_DF, threshold):
        """
        Calculate the AUC at each rank on the AUPRC curve
        """
        tmp_df = PR_CURVE_DF[PR_CURVE_DF["Threshold"] >= threshold]

        # If we only have 1 point do not calculate AUC
        if len(tmp_df["Threshold"].unique()) == 1:
            return 0
        else:
            return metrics.auc(y=tmp_df["Precision"], x=tmp_df["Recall"])


def compute_calibration_curve(goldstandard, prediction, gs_bins=None, rand_bins=None):
    """
    Build a monotonic Precision/Recall/Threshold calibration curve (plus F1) for one
    goldstandard/prediction pair. log2FC is added only when gs_bins and rand_bins are given.
    """
    precision, recall, thresholds = precision_recall_curve(goldstandard, prediction)

    P = np.maximum.accumulate(np.array(precision))
    R = np.minimum.accumulate(np.array(recall))

    # sklearn returns one fewer threshold than precision/recall; label the terminal
    # sentinel row (P=1, R=0) with the last real threshold.
    threshold_col = np.append(thresholds, thresholds[-1])
    curve_df = pd.DataFrame({'Precision': P, 'Recall': R, "Threshold": threshold_col})
    # Duplicate the last row rather than forcing Threshold=1 (quant predictions can exceed 1)
    new_row = curve_df.tail(n=1)
    curve_df = pd.concat([curve_df, new_row], ignore_index=True)

    if gs_bins is not None and rand_bins is not None:
        random_precision = gs_bins / rand_bins
        curve_df['log2FC'] = np.log2(curve_df['Precision'] / random_precision)

    curve_df['F1'] = f1_from_precision_recall(curve_df['Precision'], curve_df['Recall'])

    return curve_df


def bin_curve_by_metric(curve_df, metric):
    """
    Bin a calibration curve by one metric column (Precision/Recall/F1) into 0.01 steps,
    keeping the max-F1 row per bin. F1 is not monotonic, so a bin may be reachable at
    more than one threshold; only one row is kept.
    """
    df = curve_df.copy()

    df['_bin'] = ((df[metric].to_numpy() * 100).round().astype(int) / 100).clip(0, 1)

    binned = df.loc[df.groupby('_bin')['F1'].idxmax()].copy()
    binned = binned.rename(columns={'_bin': 'Bin'})
    binned['Metric'] = metric

    return binned[['Metric', 'Bin', 'Precision', 'Recall', 'Threshold', 'F1']].reset_index(drop=True)


def extend_bins_to_full_grid(binned_df, full_bins=None):
    """
    Extend a bin_curve_by_metric() result to every bin in full_bins (default 0.00-1.00 in
    0.01 steps). Interior gaps are linearly interpolated; exterior gaps hold the nearest
    real row. This keeps every cell type on the same grid so the cross-cell-type median
    stays monotonic.
    """
    if full_bins is None:
        full_bins = np.round(np.arange(0, 1.01, 0.01), 2)

    binned_df = binned_df.drop_duplicates(subset='Bin').sort_values('Bin').reset_index(drop=True)
    grid = pd.DataFrame({'Bin': np.sort(np.unique(full_bins))})

    extended = grid.merge(binned_df, on='Bin', how='left')
    # Row position equals Bin position on the evenly spaced grid, so linear is exact
    extended[['Precision', 'Recall', 'Threshold', 'F1']] = extended[
        ['Precision', 'Recall', 'Threshold', 'F1']
    ].interpolate(method='linear', limit_direction='both')
    extended['Metric'] = extended['Metric'].ffill().bfill()

    # Recompute F1 from interpolated P/R for internal consistency
    extended['F1'] = f1_from_precision_recall(extended['Precision'], extended['Recall'])

    return extended[['Metric', 'Bin', 'Precision', 'Recall', 'Threshold', 'F1']]


def median_bins_across_samples(binned_tables, full_bins=None):
    """
    Median of Precision/Recall/Threshold/F1 per bin across cell types for one metric.
    Each table is first extended to the full bin grid so every bin's median covers the
    same set of cell types.
    """
    metric = binned_tables[0]['Metric'].iloc[0]
    extended_tables = [extend_bins_to_full_grid(t, full_bins) for t in binned_tables]

    merged = pd.concat(extended_tables, ignore_index=True)
    combined = merged.groupby('Bin', as_index=False)[['Precision', 'Recall', 'Threshold', 'F1']].median()
    combined.insert(0, 'Metric', metric)

    return combined.sort_values('Bin').reset_index(drop=True)


def recompute_f1(median_table):
    """
    Recompute F1 from a median table's own Precision/Recall so each row is internally
    consistent (medians taken per column need not satisfy F1 = 2PR/(P+R)).
    """
    out = median_table.copy()
    out['F1'] = f1_from_precision_recall(out['Precision'], out['Recall'])
    return out


def bin_median_table_by_f1(median_tables):
    """
    Build the F1 grid by pooling already-median tables and binning by F1 (0.01 steps,
    max-F1 row per bin). Pass a single table (e.g. Recall) to keep Threshold ordering
    from one consistent source.
    """
    # Drop the stale Bin/Metric before computing the F1-based Bin
    pooled = pd.concat(median_tables, ignore_index=True).drop(columns=['Bin', 'Metric'])
    pooled['Bin'] = ((pooled['F1'].to_numpy() * 100).round().astype(int) / 100).clip(0, 1)

    binned = pooled.loc[pooled.groupby('Bin')['F1'].idxmax()].copy()
    binned['Metric'] = 'F1'

    return binned[['Metric', 'Bin', 'Precision', 'Recall', 'Threshold', 'F1']].sort_values('Bin').reset_index(drop=True)


def merge_binned_metrics(median_tables):
    """
    Concatenate the Precision/Recall/F1 tables, ordered by Metric then Bin. No
    deduplication across metrics: F1 rows legitimately coincide with Precision/Recall rows.
    """
    merged = pd.concat(median_tables, ignore_index=True)
    merged['Metric'] = pd.Categorical(merged['Metric'], categories=['Precision', 'Recall', 'F1'], ordered=True)
    merged = merged.sort_values(['Metric', 'Bin']).reset_index(drop=True)
    merged['Metric'] = merged['Metric'].astype(str)

    return merged


def build_cross_cell_type_threshold_table(curves):
    """
    Given one calibration curve per cell type: bin each by Precision and Recall, take the
    median across cell types per bin, recompute F1, derive the F1 grid from the Recall
    table, and merge the three metrics' tables.
    """
    precision_tables = [bin_curve_by_metric(curve, 'Precision') for curve in curves]
    recall_tables = [bin_curve_by_metric(curve, 'Recall') for curve in curves]

    precision_median = recompute_f1(median_bins_across_samples(precision_tables))
    recall_median = recompute_f1(median_bins_across_samples(recall_tables))

    f1_binned = bin_median_table_by_f1([recall_median])

    return merge_binned_metrics([precision_median, recall_median, f1_binned])


def sample_curve_at_thresholds(curve_df, threshold_values):
    """
    Evaluate a monotonic step-function calibration curve at a grid of thresholds using a
    forward lookup (smallest recorded threshold >= T). Out-of-range values hold the
    nearest boundary row.
    """
    curve_df = curve_df.drop_duplicates(subset='Threshold').sort_values('Threshold').reset_index(drop=True)
    shared = pd.DataFrame({'Threshold': np.sort(np.unique(threshold_values))})

    sampled = pd.merge_asof(shared, curve_df, on='Threshold', direction='forward')
    sampled = sampled.ffill().bfill()

    return sampled


def hybrid_subset(df, last_row_df_source, n_bins=300):
    """
    Downsample a threshold-calibration dataframe on a hybrid grid: Precision bins (0.01
    steps) and log-spaced Threshold bins, keeping the max-F1 row per bin.
    last_row_df_source is re-appended so the endpoint survives the downsampling.
    """
    df = df.copy()
    df = pd.concat([df, last_row_df_source], ignore_index=True)

    # Precision grid
    df['Precision_bin'] = (df['Monotonic_Precision'] * 100).round().astype(int) / 100
    df_precision = df.loc[df.groupby('Precision_bin')['Monotonic_F1'].idxmax()]

    # Threshold grid (log-spaced, denser at small values)
    threshold_bins = np.geomspace(df['Threshold'].min() + 1e-12, df['Threshold'].max(), n_bins)
    df['Threshold_bin'] = pd.cut(df['Threshold'], bins=threshold_bins)
    df_threshold = df.loc[df.groupby('Threshold_bin', observed=True)['Monotonic_F1'].idxmax()]

    # Dedupe on Threshold: df_precision lacks Threshold_bin, so full-row dedupe would miss overlaps
    df_hybrid = pd.concat([df_precision, df_threshold], ignore_index=True)
    df_hybrid = df_hybrid.drop_duplicates(subset='Threshold').sort_values('Threshold').reset_index(drop=True)
    del df_hybrid['Precision_bin']
    del df_hybrid['Threshold_bin']

    return df_hybrid
