import os
import pytest
import random

from maxatac.utilities.prepare import get_splitted_chromosomes
from maxatac.utilities.training_tools import RandomRegionsPool

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


@pytest.mark.parametrize(
    "seed, chroms, tchroms, vchroms, proportion, control_tchroms, control_vchroms",
    [
        (
            10,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]},
                "chr11": {"length": 135006516, "region": [0, 135006516]},
                "chr12": {"length": 133851895, "region": [0, 133851895]},
                "chr13": {"length": 115169878, "region": [0, 115169878]},
                "chr14": {"length": 107349540, "region": [0, 107349540]},
                "chr15": {"length": 102531392, "region": [0, 102531392]},
                "chr16": {"length": 90354753, "region": [0, 90354753]},
                "chr17": {"length": 81195210, "region": [0, 81195210]},
                "chr18": {"length": 78077248, "region": [0, 78077248]},
                "chr19": {"length": 59128983, "region": [0, 59128983]},
                "chr20": {"length": 63025520, "region": [0, 63025520]},
                "chr21": {"length": 48129895, "region": [0, 48129895]},
                "chr22": {"length": 51304566, "region": [0, 51304566]},
                "chrX": {"length": 155270560, "region": [0, 155270560]}
            },
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]}
            },
            {
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            0.5,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr20": {"length": 63025520, "region": [0, 63025520]},
                "chr11": {"length": 135006516, "region": [0, 135006516]},
                "chr17": {"length": 81195210, "region": [0, 81195210]},
                "chr18": {"length": 78077248, "region": [0, 78077248]},
                "chr22": {"length": 51304566, "region": [0, 51304566]},
                "chr14": {"length": 107349540, "region": [0, 107349540]}
            },
            {
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]},
                "chr12": {"length": 133851895, "region": [0, 133851895]},
                "chr15": {"length": 102531392, "region": [0, 102531392]},
                "chr13": {"length": 115169878, "region": [0, 115169878]},
                "chr21": {"length": 48129895, "region": [0, 48129895]},
                "chr16": {"length": 90354753, "region": [0, 90354753]},
                "chr19": {"length": 59128983, "region": [0, 59128983]},
                "chrX": {"length": 155270560, "region": [0, 155270560]}
            }
        ),
        (
            10,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            {},
            {},
            0.5,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr9": {"length": 141213431, "region": [0, 141213431]}
            },
            {
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            }
        ),
        (
            10,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]}
            },
            {},
            0.5,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            {
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr9": {"length": 141213431, "region": [0, 141213431]}
            }
        ),
        (
            10,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            {},
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]}
            },
            0.5,
            {
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr9": {"length": 141213431, "region": [0, 141213431]}
            }
        ),
        (
            10,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]}
            },
            {
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            0.5,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]}
            },
            {
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            }
        ),
        (
            10,
            {
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            {},
            {
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            0.5,
            {},
            {
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            }
        )
    ]
)
def test_get_splitted_chromosomes(seed, chroms, tchroms, vchroms, proportion, control_tchroms, control_vchroms):
    random.seed(seed)
    result_tchroms, result_vchroms = get_splitted_chromosomes(
        chroms,
        tchroms,
        vchroms,
        proportion,
    )
    assert result_tchroms == control_tchroms and result_vchroms == control_vchroms


@pytest.mark.parametrize(
    "chroms, chrom_pool_size, region_length, preferences, control_chrom_pool",
    [
        (
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            10,
            10240,
            None,
            [
                ("chr6", 171115067),
                ("chr3", 198022430),
                ("chr8", 146364022),
                ("chr2", 243199373),
                ("chr9", 141213431),
                ("chr5", 180915260),
                ("chr4", 191154276),
                ("chr7", 159138663),
                ("chr1", 249250621),
                ("chr10", 135534747)
            ]
        ),
        (
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            20,
            10240,
            None,
            [
                ("chr1", 249250621),
                ("chr1", 249250621),
                ("chr1", 249250621),
                ("chr2", 243199373),
                ("chr2", 243199373),
                ("chr2", 243199373),
                ("chr3", 198022430),
                ("chr3", 198022430),
                ("chr4", 191154276),
                ("chr4", 191154276),
                ("chr5", 180915260),
                ("chr5", 180915260),
                ("chr6", 171115067),
                ("chr6", 171115067),
                ("chr7", 159138663),
                ("chr7", 159138663),
                ("chr8", 146364022),
                ("chr8", 146364022),
                ("chr9", 141213431),
                ("chr9", 141213431),
                ("chr10", 135534747)
            ]
        ),
        (
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            5,
            10240,
            None,
            [
                ("chr1", 249250621),
                ("chr2", 243199373),
                ("chr3", 198022430),
                ("chr4", 191154276)
            ]
        ),
        (
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            10,
            700,
            os.path.join(DATA_FOLDER, "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"),
            [
                ("chr5", 180915260),
                ("chr5", 180915260),
                ("chr5", 180915260),
                ("chr5", 180915260),
                ("chr5", 180915260),
                ("chr5", 180915260),
                ("chr5", 180915260),
                ("chr5", 180915260),
                ("chr5", 180915260),
                ("chr5", 180915260)
            ]
        ),
        (
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            10,
            10,
            os.path.join(DATA_FOLDER, "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"),
            [
                ("chr6", 171115067),
                ("chr3", 198022430),
                ("chr8", 146364022),
                ("chr2", 243199373),
                ("chr9", 141213431),
                ("chr5", 180915260),
                ("chr4", 191154276),
                ("chr7", 159138663),
                ("chr1", 249250621),
                ("chr10", 135534747)
            ]
        ),
        (
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr4": {"length": 191154276, "region": [0, 191154276]},
                "chr5": {"length": 180915260, "region": [0, 180915260]},
                "chr6": {"length": 171115067, "region": [0, 171115067]},
                "chr7": {"length": 159138663, "region": [0, 159138663]},
                "chr8": {"length": 146364022, "region": [0, 146364022]},
                "chr9": {"length": 141213431, "region": [0, 141213431]},
                "chr10": {"length": 135534747, "region": [0, 135534747]}
            },
            10,
            600,
            os.path.join(DATA_FOLDER, "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"),
            [
                ("chr2", 243199373),
                ("chr1", 249250621),
                ("chr5", 180915260),
                ("chr1", 249250621),
                ("chr9", 141213431),
                ("chr2", 243199373),
                ("chr2", 243199373),
                ("chr5", 180915260),
                ("chr1", 249250621),
                ("chr9", 141213431)
            ]
        )
    ]
)
def test_random_regions_pool_chrom_pool(chroms, chrom_pool_size, region_length, preferences, control_chrom_pool):
    regions_pool = RandomRegionsPool(
        chroms=chroms, 
        chrom_pool_size=chrom_pool_size, 
        region_length=region_length,
        preferences=preferences
    )
    assert sorted(regions_pool.chrom_pool) == sorted(control_chrom_pool), \
        "Failed to build chromosome pool"


@pytest.mark.parametrize(
    "chroms, chrom_pool_size, region_length, preferences, control_preference_pool",
    [
        (
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr11": {"length": 135006516, "region": [0, 135006516]},
                "chr14": {"length": 107349540, "region": [0, 107349540]},
                "chr19": {"length": 59128983, "region": [0, 59128983]},
                "chr5": {"length": 180915260, "region": [0, 180915260]}
            },
            10,
            700,
            os.path.join(DATA_FOLDER, "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"),
            {
                "chr11": [
                    [118560389, 118561257],
                    [118560389, 118561257],
                    [119226984, 119227727],
                    [119226984, 119227727]
                ],
                "chr14": [
                    [77379582, 77380315]
                ],
                "chr19": [
                    [13075732, 13076575],
                    [13075732, 13076575],
                    [49946629, 49947351],
                    [49946629, 49947351],
                    [49946629, 49947351]
                ],
                "chr5": [
                    [142092593, 142093297],
                    [142092593, 142093297]
                ]
            }
        ),
        (
            {
                "chr14": {"length": 107349540, "region": [0, 107349540]},
                "chr19": {"length": 59128983, "region": [0, 59128983]},
                "chr5": {"length": 180915260, "region": [0, 180915260]}
            },
            10,
            700,
            os.path.join(DATA_FOLDER, "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"),
            {
                "chr14": [
                    [77379582, 77380315]
                ],
                "chr19": [
                    [13075732, 13076575],
                    [13075732, 13076575],
                    [49946629, 49947351],
                    [49946629, 49947351],
                    [49946629, 49947351]
                ],
                "chr5": [
                    [142092593, 142093297],
                    [142092593, 142093297]
                ]
            }
        ),
        (
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]}
            },
            10,
            700,
            os.path.join(DATA_FOLDER, "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"),
            {}
        ),
        (
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr11": {"length": 135006516, "region": [0, 135006516]},
                "chr14": {"length": 107349540, "region": [0, 107349540]},
                "chr19": {"length": 59128983, "region": [0, 59128983]},
                "chr5": {"length": 180915260, "region": [0, 180915260]}
            },
            10,
            700000,
            os.path.join(DATA_FOLDER, "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"),
            {}
        )
    ]
)
def test_random_regions_pool_preference_pool(
    chroms,
    chrom_pool_size,
    region_length,
    preferences,
    control_preference_pool
):
    regions_pool = RandomRegionsPool(
        chroms=chroms, 
        chrom_pool_size=chrom_pool_size, 
        region_length=region_length,
        preferences=preferences
    )
    assert control_preference_pool == regions_pool.preference_pool, \
        "Failed to build preference pool"


@pytest.mark.parametrize(
    "seed, chroms, chrom_pool_size, region_length, preferences, control_region",
    [
        (
            10,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr11": {"length": 135006516, "region": [0, 135006516]},
                "chr14": {"length": 107349540, "region": [0, 107349540]},
                "chr19": {"length": 59128983, "region": [0, 59128983]},
                "chr5": {"length": 180915260, "region": [0, 180915260]}
            },
            10,
            700,
            os.path.join(DATA_FOLDER, "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"),
            ("chr19", 49946649, 49947349)
        ),
        (
            10,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]},
                "chr11": {"length": 135006516, "region": [0, 135006516]}
            },
            10,
            700,
            os.path.join(DATA_FOLDER, "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"),
            ("chr11", 119227025, 119227725)
        ),
        (
            10,
            {
                "chr1": {"length": 249250621, "region": [0, 249250621]},
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]}
            },
            10,
            700,
            None,
            ("chr2", 175415196, 175415896)
        )
    ]
)
def test_random_regions_pool_get_region(
    seed,
    chroms,
    chrom_pool_size,
    region_length,
    preferences,
    control_region
):
    random.seed(seed)
    regions_pool = RandomRegionsPool(
        chroms=chroms, 
        chrom_pool_size=chrom_pool_size, 
        region_length=region_length,
        preferences=preferences
    )
    random_region = regions_pool.get_region()
    assert random_region == control_region, \
        "Failed to get random region"
