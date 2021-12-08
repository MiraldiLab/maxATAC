import subprocess
import pandas as pd
import subprocess
import sys

def raise_exception(e, package, install_link):
    print("command '{}' return with error (code {}): {}. Make sure {} is installed in your path {}.".format(e.cmd, e.returncode, e.output, package, install_link))
    sys.exit()
    
def check_packages_installed():
    try:
        subprocess.run(["which", "samtools"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "samtools", "http://www.htslib.org/")
        
    try:
        subprocess.run(["which", "bedtools"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "bedtools", "https://bedtools.readthedocs.io/")
        
    try:
        subprocess.run(["which", "pigz"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "pigz", "https://zlib.net/pigz/")
        
    try:
        subprocess.run(["which", "bedGraphToBigWig"], stdout=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise_exception(e, "bedGraphToBigWig", "https://anaconda.org/bioconda/ucsc-bedgraphtobigwig")
        
def convert_fragments_to_tn5_bed(fragments_tsv: str, chroms: list):
    """Convert 10X scATAC fragments file to Tn5 insertion sites bed

    Args:
        fragments_tsv (str): Path to the scATAC-seq fragments file

    Returns:
        dataframe: Tn5 insertions sites in a BED formatted pandas dataframe
        
    Examples:
    
    >>> bed_file = convert_fragments_to_tn5_bed(Granja_frags.tsv, ["chr1, "chr2"])
    """
    col_types = {
                 "chr": "category", 
                 "start": "int32", 
                 "stop": "int32", 
                 "barcode": "category", 
                 "support": "int32"
                }
    
    # Import fragments tsv as a dataframe
    df = pd.read_table(fragments_tsv,
                       header=None,
                       names=["chr", "start", "stop", "barcode", "support"],
                       dtype=col_types, 
                       low_memory=False)
    
    df = df[df["chr"].isin(chroms)]
    
    # subset starts
    df_starts = df[["chr", "start", "barcode"]].copy()
    
    # subset stops
    df_ends = df[["chr", "stop", "barcode"]].copy()
    
    # Add a 1 to create 1 bp intervals representing the cut site
    df_starts["stop"] = df_starts["start"].copy() + 1
    
    # Subtract a 1 bp interval to represent the cut site. Reference miralidlab wiki page for scATAC-seq analysis
    df_ends["start"] = df_ends["stop"].copy() - 1
    
    df_cat = pd.concat([df_starts, df_ends])

    # If split is True, return two dataframes, one for each end of the fragment    
    return df_cat[["chr", "start", "stop", "barcode"]]
