import random


def get_splitted_chromosomes(chroms,
                             tchroms,
                             vchroms,
                             proportion
                             ):
    """
    Doesn't take regions into account.
    May produce not correct results if inputs are not received from
    get_synced_chroms with ignore_regions=True
    """
    free_chrom_set = set(chroms) - set(tchroms) - set(vchroms)

    n = round(len(free_chrom_set) * proportion)

    # need sorted list for random.sample to reproduce in tests
    tchrom_set = set(random.sample(sorted(free_chrom_set), n))
    vchrom_set = free_chrom_set.difference(tchrom_set).union(set(vchroms))
    tchrom_set = tchrom_set.union(set(tchroms))

    extended_tchrom = {
        chrom_name: {
            "length": chroms[chrom_name]["length"],
            "region": chroms[chrom_name]["region"]
        } for chrom_name in tchrom_set}

    extended_vchrom = {
        chrom_name: {
            "length": chroms[chrom_name]["length"],
            "region": chroms[chrom_name]["region"]
        } for chrom_name in vchrom_set}

    return extended_tchrom, extended_vchrom
