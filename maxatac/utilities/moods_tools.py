import MOODS.scan
import MOODS.tools
import MOODS.parsers

from itertools import groupby, chain
import os
import sys
import pandas as pd
import logging


# --- Helper functions for IO ---


def pfm_to_log_odds(filename, args):
    if args.log_base is not None:
        mat = MOODS.parsers.pfm_to_log_odds(filename, args.lo_bg, args.ps, args.log_base)
    else:
        mat = MOODS.parsers.pfm_to_log_odds(filename, args.lo_bg, args.ps)
    if len(mat) != 4:
        return False, mat
    else:
        return True, mat


def adm_to_log_odds(filename, args):
    if args.log_base is not None:
        mat = MOODS.parsers.adm_to_log_odds(filename, args.lo_bg, args.ps, 4, args.log_base)
    else:
        mat = MOODS.parsers.adm_to_log_odds(filename, args.lo_bg, args.ps, 4)
    if len(mat) != 16:
        return False, mat
    else:
        return True, mat


def pfm(filename):
    mat = MOODS.parsers.pfm(filename)
    if len(mat) != 4:
        return False, mat
    else:
        return True, mat


def adm(filename):
    mat = MOODS.parsers.adm_1o_terms(filename, 4)
    if len(mat) != 16:
        return False, mat
    else:
        return True, mat


def read_text_or_fasta(filename):
    with open(filename, "r") as f:
        for line in f:
            if len(line) == 0:
                pass
            else:
                line = line.strip()
                break
    if line[0] == '>':
        return iter_fasta(filename)
    else:
        return iter_text(filename)


def iter_text(filename):
    with open(filename, "r") as f:
        seq = "".join([line.strip() for line in f])
    yield os.path.basename(filename), seq


def iter_fasta(filename):
    with open(filename, "r") as f:
        iterator = groupby(f, lambda line: line[0] == ">")
        for is_header, group in iterator:
            if is_header:
                header = next(group)[1:].strip()
            else:
                yield header, "".join(s.strip() for s in group)


# format hit sequence with variants applied
# only for for snps right now
def modified_seq(seq, start, end, applied_variants, variants):
    ret = list(seq[start:end].lower())
    for i in applied_variants:
        # print(ret)
        # print(variants[i].start_pos - start)
        ret[variants[i].start_pos - start] = (variants[i].modified_seq).upper()
    return "".join(ret)


# print results
def print_results(header, seq, matrices, matrix_names, results, args, output_target, results_snps=[], variants=[]):
    if args.no_rc:
        fr = results
        frs = results_snps
        rr = [[] for m in matrices]
        rrs = [[] for m in matrices]
    else:
        fr = results[:len(matrix_names)]
        frs = results_snps[:len(matrix_names)]
        rr = results[len(matrix_names):]
        rrs = results_snps[len(matrix_names):]

    # mix the results together, use + and - to indicate strand
    results = [[(r.pos, r.score, '+', ()) for r in fr[i]] + [(r.pos, r.score, '-', ()) for r in rr[i]] + [
        (r.pos, r.score, '+', r.variants) for r in frs[i]] + [(r.pos, r.score, '-', r.variants) for r in rrs[i]] for i
               in range(len(matrix_names))]

    for (matrix, matrix_name, result) in zip(matrices, matrix_names, results):
        if args.verbosity >= 2:
            print("{}: {}: {} matches for {}".format(os.path.basename(__file__), header, len(result), matrix_name),
                  file=sys.stderr)
        if len(matrix) == 4:
            l = len(matrix[0])
        if len(matrix) == 16:
            l = len(matrix[0]) + 1
        for r in sorted(result, key=lambda r: r[0]):
            strand = r[2]
            pos = r[0]
            hitseq = seq[pos:pos + l]
            if len(r[3]) >= 1:
                hitseq_actual = modified_seq(seq, pos, pos + l, r[3], variants)
            else:
                hitseq_actual = ""

            print("\t".join([header, matrix_name, str(pos), strand, str(r[1]), hitseq, hitseq_actual]),
                  file=output_target)


class MotifDB(object):
    def __init__(self,
                 TF: str,
                 motif_meta: str):

        self.TF = TF
        self.motif_meta = motif_meta

        self.motif_df = self.import_tf_meta()

        self.CISBP_TF = self.motif_df["TF_Name"].unique().tolist()

        if self.TF in self.CISBP_TF:
            self.tf_motif_df = self.motif_df[self.motif_df["TF_Name"] == self.TF]

            self.number_motifs = self.tf_motif_df.shape[0]

            self.motifs = self.motif_df[self.motif_df["TF_Name"] == self.TF]["Motif_Name"].tolist()

            logging.error(f"Found {self.number_motifs} motifs for {self.TF}")

        else:
            logging.error(f'No motifs for {self.TF} found')

            sys.exit(1)

    def import_tf_meta(self):
        """
        Get the motif filenames for the TF of interest

        Args:
            motif_meta: Headerless TSV file with 3 columns: 1-TF name, 2-Motif Name, 3-Source

        Returns:
            motif meta as a dataframe
        """
        # Import meta as a df
        return pd.read_table(self.motif_meta, header=None, names=["TF_Name", "Motif_Name", "Source"])


def reformat_moods_results(moods_results_path, output_bed_filename):
    column_names = ["ATAC_PEAK_ID", "TF_pos", "Strand", 'Match_Score', "Motif_sequence"]

    column_info = {
        'ATAC_PEAK_ID': "category",
        'TF_pos': "uint16",
        'Strand': "category",
        'Match_Score': "float",
        'Motif_sequence': "category"
    }

    # Holds the moods data
    moods_results = pd.read_csv(moods_results_path,
                                usecols=[0, 2, 3, 4, 5],
                                sep="\t",
                                names=column_names,
                                dtype=column_info,
                                header=None,
                                low_memory=False)

    # Get chromosome and coordinates from ATAC_PEAK_ID column
    moods_results["chr"], moods_results["Coord"] = moods_results["ATAC_PEAK_ID"].str.split(':').str

    # Get the TF start position
    moods_results["TF_start"] = moods_results["Coord"].str.split("-").str[0].apply(int) + moods_results["TF_pos"]

    # Get the TF stop position
    moods_results["TF_end"] = moods_results["TF_start"] + moods_results["Motif_sequence"].str.len()

    # Drop all columns except these listed
    moods_results = moods_results[["chr", "TF_start", "TF_end", "ATAC_PEAK_ID", "Match_Score", "Strand"]]

    # Create a dataframe of only the TF in loop, sort, drop duplicates, and write
    moods_results.sort_values(by=['chr', "TF_start", "TF_end", "Match_Score", "Strand"],
                              inplace=True)

    moods_results.drop_duplicates(subset=['chr', "TF_start", "TF_end", "Match_Score", "Strand"],
                                  keep='first',
                                  inplace=True)

    moods_results.to_csv(output_bed_filename,
                    sep="\t",
                    index=False,
                    header=False)

    return moods_results


def build_scanner(background, threshold, pval, batch, matrices_all, threshold_precision, scanner_value=7):
    scanner = MOODS.scan.Scanner(scanner_value)

    if threshold is not None:
        logging.error("Using the user defined threshold")

        thresholds = [threshold for m in matrices_all]

    if pval is not None and batch:
        logging.error("Computing thresholds from p-value for the batch of sequences (estimation)")

        thresholds = [MOODS.tools.threshold_from_p_with_precision(m, background, pval, threshold_precision, 4)
                      for m
                      in matrices_all]

    if pval is not None and not batch:
        logging.error("Computing the threshold for each sequence. (--batch == False). This will be done for each seq.")

        thresholds = [0 for m in matrices_all]

    scanner.set_motifs(matrices_all, background, thresholds)

    return thresholds, scanner, background


def scan_sequence(header,
                  seq,
                  p_val=None,
                  batch=None,
                  matrices=None,
                  matrix_names=None,
                  output_target=None,
                  threshold_precision=None,
                  matrices_all=None,
                  no_snps=None,
                  args=None,
                  scanner=None):
    # preprocessing for the new sequence if using p-values and not --batch
    if p_val is not None and not batch:
        bg = MOODS.tools.bg_from_sequence_dna(seq, 1)

        logging.debug(f"Estimated background for {header} is {bg}")

        thresholds = [
            MOODS.tools.threshold_from_p_with_precision(m, bg, p_val, threshold_precision, 4) for
            m in
            matrices_all]

        logging.debug("Preprocessing matrices for scanning")
        scanner = MOODS.scan.Scanner(7)
        scanner.set_motifs(matrices_all, bg, thresholds)

    results = scanner.scan(seq)

    #print(results)
    if no_snps:
        results_snps = [[]] * len(matrices_all)
        snps = []

    else:
        snps = MOODS.tools.snp_variants(seq)

        results_snps = scanner.variant_matches(seq, snps)

    print_results(header, seq, matrices, matrix_names, results, args, output_target, results_snps, snps)
