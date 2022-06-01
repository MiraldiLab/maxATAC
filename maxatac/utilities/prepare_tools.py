import pandas as pd

def convert_fragments_to_tn5_bed(fragments_tsv: str, chroms: list):
    """Convert 10X scATAC fragments file to Tn5 insertion sites bed

    Args:
        fragments_tsv (str): Path to the scATAC-seq fragments file

    Returns:
        dataframe: Tn5 insertions sites in a BED formatted pandas dataframe
        
    Examples:
    
    >>> bed_file = convert_fragments_to_tn5_bed(Granja_frags.tsv, ["chr1", "chr2"])
    """
    
    # Import fragments tsv as a dataframe
    df = pd.read_table(fragments_tsv,
                       sep="\t",
                       header=None,
                       usecols=[0,1,2,3],
                       names=["chr", "start", "stop", "barcode"]
                       )
    
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
