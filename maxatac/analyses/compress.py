import pyBigWig
import numpy as np
import os
import timeit
import logging
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.genome_tools import build_chrom_sizes_dict

def change_bigWig_compression_level(args):
    """
    Generates (un)compressed bigWig files using the --max_zooms parameter option in pyBigWig.

    Args:
        input_bigWig (str): The path to a bigWig file that should be compressed with this function.
        chrom_sizes (dict): The path to a dictionary of chromosome sizes used to form the output bigWig file
        chroms (list): A list of chromosomes that should be represented in the output bigWig file
        prefix (str): The output bigWig filename
        max_zooms (int): A value between 0 and 10 that represents how many zoom levels should be computed 
        (e.g., for visualization) for the output bigWig file. A value of 0 results in the greatest compression,
        but prevents visualization of the output file. A value of 10 promotes fast visualization of the output
        file but is storage-intensive.

    Returns:
        An output bigWig file with the user-specified number of zoom levels.
    """
    
    input_bigWig = args.input_bigWig
    chromosomes = args.chroms
    chromosome_length_dictionary = build_chrom_sizes_dict(chromosomes, args.chrom_sizes)
    output_dir = get_dir(args.output_dir)
    output_filename = args.prefix
    max_zooms = args.max_zooms
    
    # Create the output path.
    output_path = os.path.join(output_dir, output_filename + ".bw")
    
    # Start a timer.
    start_time = timeit.default_timer()
    
    # Output the starting size of the file in Mb.
    input_file_size = os.path.getsize(input_bigWig) / (1024 ** 2)
    logging.info("Input file size: %d Mb.\n" % (input_file_size))
    
    with pyBigWig.open(input_bigWig) as input_bigWig, pyBigWig.open(output_path, "w") as output_bigWig:
        # Make the bigWig header using the specified chrom sizes dictionary and chromosomes, then add it to 
        # the output bigWig file with the user-specified maxZooms level.
        logging.info("Now writing the header for the bigWig file.")
        header = [(x, chromosome_length_dictionary[x]) for x in sorted(chromosomes)]
        output_bigWig.addHeader(header, maxZooms = max_zooms)

        # For every chromosome in the header, add the corresponding signal values from the input bigWig file
        # as quantized float16 values for a slightly higher amount of compression.
        for chromosome, chromosome_length in header:
            logging.info("Now adding bigWig entries for: " + chromosome)
            signal_values = np.nan_to_num(input_bigWig.values(chromosome, 0, chromosome_length, numpy = True))            
            output_bigWig.addEntries(chroms = chromosome,
                                     starts = 0,
                                     ends = chromosome_length,
                                     span = 1,
                                     step = 1,
                                     values = signal_values.astype(np.float16).tolist()
                                     )
        
        # Measure the amount of time required for bigWig (de)compression. Output it in a nice format.
        stop_time = timeit.default_timer()
        mins, secs = divmod(stop_time - start_time, 60)
        hours, mins = divmod(mins, 60)

        logging.info("Total (de)compression time: %d:%d:%d.\n" % (hours, mins, secs))
       
        # Output the size of the output file in Mb.
        output_file_size = os.path.getsize(output_path) / (1024 ** 2)
        logging.info("Output file size: %d Mb.\n" % (output_file_size)) 
        logging.info("The output bigWig has been saved to: " + output_path)
        