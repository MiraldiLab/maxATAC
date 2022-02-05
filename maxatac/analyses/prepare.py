"""Run maxatac prepare for generating normalized signal tracks for prediction
"""
import logging
import sys
import subprocess
import os
import pysam
from maxatac.utilities.system_tools import get_dir, check_prepare_packages_installed
from maxatac.utilities.prepare_tools import convert_fragments_to_tn5_bed
from maxatac.utilities.constants import ALL_CHRS, PREPARE_scATAC_SCRIPT, PREPARE_BULK_SCRIPT
from maxatac.analyses.normalize import run_normalization


def run_prepare(args):
    """Run maxatac prepare for generating normalized signal tracks for prediction
    
    This function will :
    1) Check whether the file is a .tsv/.tsv.gz or .bam file. 
    
        tsv file: 10x scATACseq fragments file
        bam file: bulk ATACseq BAM file
            If the file is a bam file, a flagstat check for duplicate reads will be performed.
        
    2) Convert reads or fragments to Tn5 cut sites and sequencing depth normalize.
    3) Minmax normalize the RPM normalized bigwig files for maxATAC    
    
    Args:
        args (argslist)): The argsparse object with input parameters as attributes.
    """
    # Check if samtools, bedtools, bedgraphtobigwig, and pigz are installed
    check_prepare_packages_installed()
    
    logging.error(f"Input file: {args.input} \n" +
                  f"Input chromosome sizes file: {args.chrom_sizes} \n" +
                  f"Tn5 cut sites will be slopped {args.slop} bps on each side \n" +
                  f"Input blacklist file: {args.blacklist} \n" +
                  f"Output filename: {args.prefix} \n" +
                  f"Output directory: {args.output} \n" +
                  f"Using a millions factor of: {args.rpm_factor} \n" +
                  f"Using {args.threads} threads to run job.")
    
    output_dir = get_dir(args.output)
    
    logging.error("Generate the normalized signal tracks.")

    if args.input.endswith(".bam"):
        logging.error("Working on a bulk ATAC-seq BAM file \n" + "Getting the number of reads in the BAM file")
        
        # Get the read count using pysam
        read_counts = int(pysam.view("-c", "-F", "260", args.input))
        
        print(f"There are {read_counts} reads in the file")
        # The scale factor is used to normalize the reads to RP20M
        # We want to divide our count by the total number of reads
        # Then we want to multiply by 20,000,000. This will give you the 
        # Number of counts normalized for sequencing depth of 20,000,000 reads.
        scale_factor = (1/read_counts) * args.rpm_factor
        
        if args.skip_dedup:
            logging.error("Processing BAM to bigwig. Skipping deduplication")
 
            # Use subprocess to run bedtools and bedgraphtobigwig
            subprocess.run(["bash", 
                            PREPARE_BULK_SCRIPT,
                            args.input, 
                            args.prefix,
                            output_dir,
                            str(args.threads),
                            args.blacklist_bed,
                            args.chrom_sizes,
                            str(args.slop), 
                            str(scale_factor),
                            "skip"], check=True)
        else:
            logging.error("Processing BAM to bigwig. Running eduplication")

            subprocess.run(["bash", 
                            PREPARE_BULK_SCRIPT,
                            args.input, 
                            args.prefix,
                            output_dir,
                            str(args.threads),
                            args.blacklist_bed,
                            args.chrom_sizes,
                            str(args.slop), 
                            str(scale_factor),
                            "deduplicate"], check=True)
                        
    elif args.input.endswith((".tsv", ".tsv.gz")):
        logging.error("Working on 10X scATAC fragments file \n " + "Converting fragment files to Tn5 sites")

        # Convert a 10X fragments file to Tn5 cut sites
        bed_df = convert_fragments_to_tn5_bed(args.input, ALL_CHRS)
        
        logging.error("Getting the number of Tn5 cut sites in the fragment file")

        # Get the counts of Tn5 cut sites for normalization
        counts = bed_df.shape[0]
        
        print(f"There are {counts} Tn5 cut sites in the file")

        # Calculate the scale factor for bedtools. This value is multiplied by the counts
        scale_factor = (1/counts) * args.rpm_factor
        
        # Create a temp file for the cut site, We will slop and save that file instead
        tmp_file_path=os.path.join(output_dir, args.prefix + "_CutSites.bed")
        
        bed_df.to_csv(tmp_file_path, sep="\t", header=False, index=False)

        logging.error("Slopping Tn5 cut sites and generating RPM normalized bigwig")
        
        # Use subprocess to run bedtools and bedgraphtobigwig
        subprocess.run(["bash", 
                        PREPARE_scATAC_SCRIPT, 
                        tmp_file_path, 
                        args.chrom_sizes, 
                        str(args.slop), 
                        args.blacklist_bed,
                        args.prefix,
                        output_dir,
                        str(scale_factor)], check=True)
                                
    else:
        print("You have not specific a correct input file type. Options: bulk or scatac")
        sys.exit()

    logging.error("Min-max normalize signal tracks")
    
    # Set the argument names to match those that are expected by run_normalization
    # TODO find a better way to implement the normalization from inside this script
    args.signal = os.path.join(output_dir, 
                               f"{args.prefix}_IS_slop{args.slop}_RP20M.bw")
    
    args.prefix = f"{args.prefix}_IS_slop{args.slop}_RP20M_minmax01"
    args.method = "min-max"
    args.min = 0
    args.max = False
    args.max_percentile = 99
    args.clip = False
    
    # Minmax normalize signal tracks
    run_normalization(args)