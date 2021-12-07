import logging
from maxatac.utilities.system_tools import get_dir, Mute
import subprocess
import pysam
from maxatac.analyses.normalize import run_normalization
import os

def run_prepare(args):
    logging.error(f"Input bam file: {args.input} \n" +
                  f"Input chromosome sizes file: {args.chrom_sizes} \n" +
                  f"Tn5 cut sites will be slopped {args.slop} bps on each side \n" +
                  f"Input blacklist file: {args.blacklist} \n" +
                  f"Output filename: {args.prefix} \n" +
                  f"Output directory: {args.output} \n" +
                  f"Using a millions factor of: {args.rpm_factor}")
    
    output_dir = get_dir(args.output)
    
    logging.error("Getting the number of reads in the BAM file")
    
    read_counts = int(pysam.view("-c", "-F", "260", args.input))
    
    print(f"There are {read_counts} reads in the file")
    # The scale factor is used to normalize the reads to RP20M
    # We want to divide our count by the total number of reads
    # Then we want to multiply by 20,000,000. This will give you the 
    # Number of counts normalized for sequencing depth of 20,000,000 reads.
    scale_factor = (1/read_counts) * args.rpm_factor
    
    logging.error("Generate the normalized signal tracks.")

    subprocess.run(["bash", 
                    os.path.join(os.path.dirname(__file__), "../../scripts/shift_reads.sh"), 
                    args.input, 
                    args.chrom_sizes, 
                    str(args.slop), 
                    args.blacklist_bed,
                    args.prefix,
                    output_dir,
                    str(scale_factor)])

    logging.error("Min-max normalize signal tracks")
    
    args.signal = os.path.join(output_dir, 
                               f"{args.prefix}_IS_slop{args.slop}_RP20M.bw")
    
    args.prefix = f"{args.prefix}_IS_slop{args.slop}_RP20M_minmax01"
    args.method = "min-max"
    args.min = 0
    args.max = False
    args.max_percentile = 99
    args.clip = False
    
    run_normalization(args)