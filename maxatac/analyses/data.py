import shutil
import os
import subprocess
import logging
from maxatac.utilities.prepare_tools import raise_exception

def check_packages_installed():
    """Check that packages are installed
    This module requires 
    """
    try:
        subprocess.run(["which", "git"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "git", "conda install git")
        
    try:
        subprocess.run(["which", "wget"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "wget", "conda install wget")
        
def run_data(args):
    """Download maxATAC reference data

    Args:
        args (Namespace): Arguments list for the reference genome and the log level. 
    """  
    check_packages_installed()
    ############### Parameters ###############
    # Path to the directory where the project data should be installed
    maxatac_original_dir = os.path.join(args.output, "maxATAC_data")
    maxatac_final_dir = os.path.join(args.output, "data")

    # Command to clone the data repo
    clone = "git clone https://github.com/MiraldiLab/maxATAC_data.git"
    wget_2bit = f"wget https://hgdownload.cse.ucsc.edu/goldenpath/{args.genome}/bigZips/{args.genome}.2bit"

    ############### Body ###############
    logging.error(f"Downloading data for: {args.genome} \n" +
                  f"Data will be installed: {args.output} \n" +
                  f"Temporarily downlading data to: {maxatac_original_dir} \n" +
                  f"Final data will be placed in: {maxatac_final_dir}")

    # Create the project directory if it does not exist
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    os.chdir(args.output) # Specifying the path where the cloned project needs to be copied
    os.system(clone) # Cloning

    dest = shutil.move(maxatac_original_dir, maxatac_final_dir) # Rename the git directory to data

    os.chdir(os.path.join(maxatac_final_dir, args.genome)) # change to the final data directory
    os.system(wget_2bit) # Wget hg38.2bit

    logging.error("Finished!")
