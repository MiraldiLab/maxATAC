import shutil
import os
import logging
from maxatac.utilities.system_tools import get_dir, check_data_packages_installed
 
def run_data(args):
    """Download maxATAC reference data

    Args:
        args (Namespace): Arguments list for the reference genome and the log level. 
    """  
    check_data_packages_installed()
    
    ############### Parameters ###############
    # Path to the directory where the project data should be installed. Default: ~/opt
    base_dir = get_dir(args.output)
    
    # Path to the maxATAC directory in the install directory. Default: ~/opt/maxatac
    maxatac_dir = get_dir(os.path.join(base_dir, "maxatac"))
    
    # Path the repo directory and the final directory. Default: ~/opt/maxatac/maxATAC_data
    maxatac_repo_dir = os.path.join(maxatac_dir, "maxATAC_data")
    
    # Path final directory (it is renamed for simplicity). Default: ~/opt/maxatac/data
    maxatac_final_dir = os.path.join(maxatac_dir, "data")

    # Command to clone the data repo
    clone = "git clone https://github.com/MiraldiLab/maxATAC_data.git"
    wget_2bit = f"wget --no-check-certificate https://hgdownload.cse.ucsc.edu/goldenpath/{args.genome}/bigZips/{args.genome}.2bit"

    ############### Body ###############
    logging.error(f"Downloading data for: {args.genome} \n" +
                  f"Data will be installed: {args.output} \n" +
                  f"Temporarily downlading data to: {maxatac_repo_dir} \n" +
                  f"Final data will be placed in: {maxatac_final_dir}")

    os.chdir(maxatac_dir) # Change to the directory where the repo should be cloned
    
    os.system(clone) # Cloning

    dest = shutil.move(maxatac_repo_dir, maxatac_final_dir) # Rename the git directory to data

    os.chdir(os.path.join(maxatac_final_dir, args.genome)) # change to the final data directory
    os.system(wget_2bit) # Wget hg38.2bit

    logging.error("Finished!")
