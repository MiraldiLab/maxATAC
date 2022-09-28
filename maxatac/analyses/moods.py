#!/usr/bin/env python

"""
Use MOODS to make transcription factor binding predictions with CISBP motifs.

This is a copy of the file https://github.com/jhkorhonen/MOODS/blob/master/python/scripts/moods-dna.py from the MOODS
github repo. I added in maxATAC specific imports and some helper functions so it would work with maxATAC. This file will
output a bedgraph file, bigwig, raw moods output, and a reformatted bed file of moods predictions.

TODO: Look into the contributions/usage policies of MOODS
"""

import logging
import os
import sys
import pandas as pd
import pybedtools
from multiprocessing import Pool
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

from maxatac.utilities.prediction_tools import write_predictions_to_bigwig
from maxatac.utilities.genome_tools import build_chrom_sizes_dict
from maxatac.utilities.moods_tools import pfm_to_log_odds, adm_to_log_odds, MotifDB, print_results, read_text_or_fasta, \
    adm, pfm, reformat_moods_results, build_scanner, scan_sequence
from maxatac.utilities.system_tools import get_dir, Mute, check_moods_packages_installed
from maxatac.utilities.constants import cisbp_motifs_meta, motifs_dir

import MOODS.scan
import MOODS.tools
import MOODS.parsers


def run_moods(args):
    #############################################
    # Parameters
    #############################################

    # TODO Check if MOODS is installed before running. Also check the motif db
    check_moods_packages_installed()

    if args.t is not None and args.batch:
        logging.error("Warning: --batch is redundant when used with -t")

    if args.log_base is not None and (args.log_base <= 1):
        logging.error("Error: --log-base has to be > 1")
        sys.exit(1)

    # Get outdir
    outdir = get_dir(args.output_dir)

    # Create output filenames
    moods_output_filename = os.path.join(outdir, args.prefix + ".moods")
    reformat_moods_filename = os.path.join(outdir, args.prefix + ".bed")
    moods_bedgraph_filename = os.path.join(outdir, args.prefix + ".bg")
    moods_bigwig_filename = os.path.join(outdir, args.prefix + ".bw")

    # Open output file for writing
    try:
        output_target = open(moods_output_filename, 'w')

    except:
        logging.error(f"Could not open output file {moods_output_filename} for writing")
        sys.exit(1)

    #############################################
    # Collect, format, and parse motifs
    #############################################
    # Get the motif files from maxATAC_data if the user specified the TF name
    if args.TF:
        # Create a motif DB object: just parses meta file for specific TF
        motif_db = MotifDB(args.TF, cisbp_motifs_meta)

        # Get the name of the motifs for the TF
        matrix_names = motif_db.motifs

        # Get the number of motifs for the TF of interest
        number_motifs = motif_db.number_motifs

        # Assign the matrix files to the args.matrix_files variable
        args.matrix_files = [os.path.join(motifs_dir, file) for file in matrix_names]

    else:
        # Parse the input motif files
        matrix_names = [os.path.basename(n) for n in args.matrix_files + args.lo_matrix_files]
        number_motifs = len(matrix_names)

    # Create empty lists to collect the formatted matrices
    matrices = []
    matrices_rc = []

    logging.error(f"Reading {number_motifs} matrix files")

    # Parse pfm count matrices
    for filename in args.matrix_files:
        valid = False
        if filename[-4:] != '.adm':  # let's see if it's pfm
            valid, matrix = pfm_to_log_odds(filename, args)

        if not valid:  # well maybe it is adm
            valid, matrix = adm_to_log_odds(filename, args)

        if not valid:
            logging.error("error: could not parse matrix file {}".format(filename))
            sys.exit(1)

        else:
            matrices.append(matrix)
            matrices_rc.append(MOODS.tools.reverse_complement(matrix, 4))

    # Parse log odds count matrices
    for filename in args.lo_matrix_files:
        valid = False
        if filename[-4:] != '.adm':  # let's see if it's pfm
            valid, matrix = pfm(filename)

        if not valid:  # well maybe it is adm
            valid, matrix = adm(filename)

        if not valid:
            logging.error("Could not parse matrix file {}".format(filename))
            sys.exit(1)

        else:
            matrices.append(matrix)
            matrices_rc.append(MOODS.tools.reverse_complement(matrix, 4))

    if args.no_rc:
        matrices_all = matrices

    else:
        matrices_all = matrices + matrices_rc

    #############################################
    # Scanning
    #############################################
    # Get scanner
    thresholds, scanner, background = build_scanner(background=args.bg,
                                                    threshold=args.t,
                                                    pval=args.p_val,
                                                    batch=args.batch,
                                                    matrices_all=matrices_all,
                                                    threshold_precision=args.threshold_precision)

    # Create an empty list to catch matches
    matches = []

    for seq_file in args.sequence_files:
        logging.error(f"Reading sequence file {seq_file}")

        try:
            seq_iterator = read_text_or_fasta(seq_file)

        except Exception:
            logging.error(f"Error: could not parse sequence file {seq_file}")
            sys.exit(1)

        for header, seq in seq_iterator:
            scan_sequence(header,
                          seq,
                          args.p_val,
                          args.batch,
                          matrices,
                          matrix_names,
                          output_target,
                          args.threshold_precision,
                          matrices_all,
                          args.no_snps,
                          args,
                          scanner)

    # Close the MOODS output file
    output_target.close()

    #############################################
    # Reformat Moods output to BED format
    #############################################
    # Reformat the original moods results so that they are deduplicated and also in BED format.
    moods_results = reformat_moods_results(moods_output_filename, reformat_moods_filename)

    #############################################
    # Convert MOODS BED to Bigwig
    #############################################
    # Convert MOODS results to bigwig of motif occurences per bp
    # Create bedtool from dataframe
    moods_bedtool = pybedtools.BedTool.from_dataframe(moods_results)

    # Get bedgraph format coverage of motifs
    moods_bedgraph = moods_bedtool.genome_coverage(bg=True, genome='hg38')

    # Convert the bedtool to a dataframe
    moods_bedgraph_df = pybedtools.BedTool.to_dataframe(moods_bedgraph)

    # Output the bedgraph formatted file
    moods_bedgraph_df.to_csv(moods_bedgraph_filename, sep="\t", header=False, index=False)

    # Change the column names to be compatible with downstream functions
    moods_bedgraph_df.columns = ["chr", "start", "stop", "score"]

    # Get the unique chromosomes that are in the predictions list
    moods_bedgraph_df_chroms = moods_bedgraph_df["chr"].unique().tolist()

    # Write predictions to bigwig
    write_predictions_to_bigwig(moods_bedgraph_df,
                                output_filename=moods_bigwig_filename,
                                chrom_sizes_dictionary=build_chrom_sizes_dict(moods_bedgraph_df_chroms,
                                                                              args.chromosome_sizes),
                                chromosomes=moods_bedgraph_df_chroms
                                )
    #############################################
    # Compress output files
    #############################################
    # Compress files
    os.system(f"pigz {moods_output_filename}")
    os.system(f"pigz {moods_bedgraph_filename}")
    os.system(f"pigz {reformat_moods_filename}")

    logging.error("Done!")
