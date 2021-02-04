import pytest
import os

from maxatac.utilities.helpers import get_files
from maxatac.utilities.parser import (
    get_synced_chroms,
    parse_arguments
)


DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


@pytest.mark.parametrize(
    "chroms, filenames, ignore_regions, control_synced_chroms",
    [
        #  correct inputs without regions
        (
            [],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {}
        ),
        (
            [],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {}
        ),
        (
            ["chr1"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        
        #  correct inputs with regions
        (
            ["chr1:10-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [10, 100]}}
        ),
        (
            ["chr1:10-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [10, 100]}}
        ),
        (
            ["chr1:10-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [10, 100]}}
        ),
        (
            ["chr1:10-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [10, 100]}}
        ),
        (
            ["chr1:10-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:10-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:10-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [10, 249250621]}}
        ),
        (
            ["chr1:10-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [10, 249250621]}}
        ),
        (
            ["chr1:10-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [10, 249250621]}}
        ),
        (
            ["chr1:10-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [10, 249250621]}}
        ),
        (
            ["chr1:10-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:10-249250621"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),

        #  not correct inputs without regions
        (
            ["chrA"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {}
        ),
        (
            ["chrA"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {}
        ),
        (
            ["chrA"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {}
        ),
        (
            ["chrA"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {}
        ),
        (
            ["chrA"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {}
        ),
        (
            ["chrA"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {}
        ),

        #  not correct inputs with regions
        (
            ["chr1:"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:-100-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:-100-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:-100-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:-100-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:-100-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:-100-100"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:100-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:100-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:100-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:100-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:100-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:100-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-500000000"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-500000000"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-500000000"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-500000000"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),        
        (
            ["chr1:0-500000000"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-500000000"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-0"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-0"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-0"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-0"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-0"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:0-0"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:10-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:10-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:10-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:10-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            False,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:10-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),
        (
            ["chr1:10-10"],
            [
                "hg19.2bit",
                "predict_signal_cell_GM12878.bigwig",
                "average.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {"chr1": {"length": 249250621, "region": [0, 249250621]}}
        ),

        # combination of mutiple files
        (
            ["chr1"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            None,
            {}
        ),
        (
            ["chr1"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {}
        ),
        (
            ["chr2"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            None,
            {"chr2": {"length": 243199373, "region": [0, 243199373]}}
        ),
        (
            ["chr2"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {"chr2": {"length": 243199373, "region": [0, 243199373]}}
        ),
        (
            ["chr2", "chr3:10-100"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            None,
            {
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [10, 100]}
            }
        ),
        (
            ["chr2", "chr3:10-100"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [10, 100]}
            }
        ),
        (
            ["chr2", "chr3:10-100"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            True,
            {
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]}
            }
        ),
        (
            ["chr2", "chr3:10-100"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]}
            }
        ),
        (
            ["chr1", "chr2", "chr3:10-100"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            None,
            {
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [10, 100]}
            }
        ),
        (
            ["chr1", "chr2", "chr3:10-100"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [10, 100]}
            }
        ),
        (
            ["chr1", "chr2", "chr3:10-100"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            True,
            {
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]}
            }
        ),
        (
            ["chr1", "chr2", "chr3:10-100"],
            [
                "hg19.2bit",
                "average.bigwig",
                "train_signal_cell_A549.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            True,
            {
                "chr2": {"length": 243199373, "region": [0, 243199373]},
                "chr3": {"length": 198022430, "region": [0, 198022430]}
            }
        ),
        (
            ["chr1"],
            [
                "validate_sites_cell_HCT116_tf_CTCF.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            None,
            {}
        ),
        (
            ["chr1"],
            [
                "validate_sites_cell_HCT116_tf_CTCF.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig",
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {}
        ),
        (
            ["chr1"],
            [
                "validate_sites_cell_HCT116_tf_CTCF.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig",
                None
            ],
            None,
            {}
        ),
        (
            ["chr1"],
            [
                "validate_sites_cell_HCT116_tf_CTCF.bigwig",
                "train_sites_cell_A549_tf_CTCF.bigwig",
                None,
                "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
            ],
            None,
            {}
        )
    ]
)
def test_get_synced_chroms(chroms, filenames, ignore_regions, control_synced_chroms):
    locations = list(get_files(DATA_FOLDER, "|".join([f for f in filenames if f is not None])).values())
    locations += [f for f in filenames if f is None]
    synced_chroms = get_synced_chroms(chroms, locations, ignore_regions)
    assert synced_chroms == control_synced_chroms


# For not direct testing of assert_and_fix_args_for_training

@pytest.mark.parametrize(
    "args, control_args",
    [
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--chroms", "chr1", "chr2", "chr3", "chr10", "chr11", "chr8", "chr21",
                "--tchroms", "chr1", "chr10", "chr11",
                "--vchroms", "chr2", "chr8", "chr21",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "validate_sites_cell_HCT116_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]},
                    "chr8": {"length": 146364022, "region": [0, 146364022]},
                    "chr21": {"length": 48129895, "region": [0, 48129895]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]}
                },
                "vchroms": {
                    "chr8": {"length": 146364022, "region": [0, 146364022]},
                    "chr21": {"length": 48129895, "region": [0, 48129895]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--chroms", "chr1", "chr2", "chr3", "chr10", "chr11", "chr8", "chr21",
                "--tchroms", "chr1", "chr10", "chr11",
                "--vchroms", "chr2", "chr8", "chr21",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "validate_sites_cell_HCT116_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]},
                    "chr8": {"length": 146364022, "region": [0, 146364022]},
                    "chr21": {"length": 48129895, "region": [0, 48129895]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]}
                },
                "vchroms": {
                    "chr8": {"length": 146364022, "region": [0, 146364022]},
                    "chr21": {"length": 48129895, "region": [0, 48129895]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--chroms", "chr1", "chr2", "chr3", "chr10", "chr11", "chr8", "chr21",
                "--tchroms", "chr10",
                "--vchroms", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "validate_sites_cell_HCT116_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]}
                },
                "vchroms": {
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--chroms", "chr1", "chr2", "chr3", "chr10", "chr11", "chr8", "chr21",
                "--tchroms", "chr10",
                "--vchroms", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "validate_sites_cell_HCT116_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]}
                },
                "vchroms": {
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--chroms", "chr1", "chr2", "chr3", "chr10", "chr11", "chr8", "chr21",
                "--vchroms", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "validate_sites_cell_HCT116_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                },
                "tchroms": {},
                "vchroms": {
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--chroms", "chr1", "chr2", "chr3", "chr10", "chr11", "chr8", "chr21",
                "--vchroms", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "validate_sites_cell_HCT116_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                },
                "tchroms": {},
                "vchroms": {
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--tchroms", "chr10",
                "--vchroms", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "validate_sites_cell_HCT116_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]}
                },
                "vchroms": {
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--tchroms", "chr10",
                "--vchroms", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "validate_sites_cell_HCT116_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]}
                },
                "vchroms": {
                    "chr8": {"length": 146364022, "region": [0, 146364022]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--tchroms", "chr1", "chr10", "chr11",
                "--vchroms", "chr8", "chr12", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr2": {"length": 243199373, "region": [0, 243199373]},
                    "chr3": {"length": 198022430, "region": [0, 198022430]},
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]},
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]}
                },
                "vchroms": {
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--tchroms", "chr1", "chr10", "chr11",
                "--vchroms", "chr8", "chr12", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr2": {"length": 243199373, "region": [0, 243199373]},
                    "chr3": {"length": 198022430, "region": [0, 198022430]},
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]},
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]}
                },
                "vchroms": {
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--vchroms", "chr8", "chr12", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr2": {"length": 243199373, "region": [0, 243199373]},
                    "chr3": {"length": 198022430, "region": [0, 198022430]},
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]},
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                },
                "tchroms": {},
                "vchroms": {
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--vchroms", "chr8", "chr12", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr2": {"length": 243199373, "region": [0, 243199373]},
                    "chr3": {"length": 198022430, "region": [0, 198022430]},
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]},
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                },
                "tchroms": {},
                "vchroms": {
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--filters", "average.bigwig",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--tchroms", "chr1", "chr10", "chr11",
                "--vchroms", "chr8", "chr12", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr2": {"length": 243199373, "region": [0, 243199373]},
                    "chr3": {"length": 198022430, "region": [0, 198022430]},
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]},
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]}
                },
                "vchroms": {
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--filters", "average.bigwig",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--tchroms", "chr1", "chr10", "chr11",
                "--vchroms", "chr8", "chr12", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr2": {"length": 243199373, "region": [0, 243199373]},
                    "chr3": {"length": 198022430, "region": [0, 198022430]},
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]},
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]}
                },
                "vchroms": {
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--filters", "train_sites_cell_A549_tf_CTCF.bigwig",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--tchroms", "chr1", "chr10", "chr11",
                "--vchroms", "chr8", "chr12", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr2": {"length": 243199373, "region": [0, 243199373]},
                    "chr3": {"length": 198022430, "region": [0, 198022430]},
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]},
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]}
                },
                "vchroms": {
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--filters", "train_sites_cell_A549_tf_CTCF.bigwig",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--tchroms", "chr1", "chr10", "chr11",
                "--vchroms", "chr8", "chr12", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr2": {"length": 243199373, "region": [0, 243199373]},
                    "chr3": {"length": 198022430, "region": [0, 198022430]},
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]},
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]},
                    "chr11": {"length": 135006516, "region": [0, 135006516]}
                },
                "vchroms": {
                    "chr12": {"length": 133851895, "region": [0, 133851895]},
                    "chr13": {"length": 115169878, "region": [0, 115169878]}
                }
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--filters", "train_sites_cell_A549_tf_CTCF.bigwig",
                "--tchroms", "chr10",
                "--vchroms", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "validate_sites_cell_HCT116_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]}
                },
                "vchroms": {}
            }
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--filters", "train_sites_cell_A549_tf_CTCF.bigwig",
                "--tchroms", "chr10",
                "--vchroms", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "validate_sites_cell_HCT116_tf_CTCF.bigwig"
            ],
            {
                "chroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]}
                },
                "tchroms": {
                    "chr10": {"length": 135534747, "region": [0, 135534747]}
                },
                "vchroms": {}
            }
        )
    ]
)
def test_parse_chromosome_arguments_for_training(args, control_args):
    args = parse_arguments(args, DATA_FOLDER)
    assert args.chroms == control_args["chroms"] and args.tchroms == control_args["tchroms"] and args.vchroms == control_args["vchroms"]


# For not direct testing of assert_and_fix_args_for_training (should fail on assert)

@pytest.mark.parametrize(
    "args",
    [
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--tchroms", "chr1", "chr10", "chr11",
                "--vchroms", "chr8", "chr10", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ]
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--tchroms", "chr1", "chr10", "chr11",
                "--vchroms", "chr8", "chr10", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ]
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--tchroms", "chr1", "chr20", "chr11",
                "--vchroms", "chr8", "chr10", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ]
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--chroms", "chr2", "chr3", "chr1", "chr10", "chr11", "chr8", "chr12", "chr13",
                "--tchroms", "chr1", "chr20", "chr11",
                "--vchroms", "chr8", "chr10", "chr13",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ]
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--chroms", "chr1", "chr8",
                "--tchroms", "chr1",
                "--vchroms", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ]
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--chroms", "chr1", "chr8",
                "--tchroms", "chr1",
                "--vchroms", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ]
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--sequence", "hg19.2bit",
                "--chroms", "chr1", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ]
        ),
        (
            [
                "train",
                "--signal", "train_signal_cell_A549.bigwig",
                "--validation", "validate_signal_cell_HCT116.bigwig",
                "--average", "average.bigwig",
                "--preferences", "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed",
                "--sequence", "hg19.2bit",
                "--chroms", "chr1", "chr8",
                "--tsites",  "train_sites_cell_A549_tf_CTCF.bigwig",
                "--vsites", "train_sites_cell_A549_tf_CTCF.bigwig"
            ]
        )
    ]
)
def test_parse_chromosome_arguments_for_training_should_fail(args):
    with pytest.raises(AssertionError):
        args = parse_arguments(args, DATA_FOLDER)
    



# average.bigwig
# chr1 249250621
# chr2 243199373
# chr3 198022430
# chr4 191154276
# chr5 180915260
# chr6 171115067
# chr7 159138663
# chr8 146364022
# chr9 141213431
# chr10 135534747
# chr11 135006516
# chr12 133851895
# chr13 115169878
# chr14 107349540
# chr15 102531392
# chr16 90354753)
# chr17 81195210
# chr18 78077248
# chr19 59128983
# chr20 63025520
# chr21 48129895
# chr22 51304566
# chrX 155270560


# predict_signal_cell_GM12878.bigwig
# chr1 249250621
# chr2 243199373
# chr3 198022430
# chr4 191154276
# chr5 180915260
# chr6 171115067
# chr7 159138663
# chr8 146364022
# chr9 141213431
# chr10 135534747)
# chr11 135006516
# chr12 133851895
# chr13 115169878
# chr14 107349540
# chr15 102531392
# chr16 90354753
# chr17 81195210
# chr18 78077248
# chr19 59128983
# chr20 63025520
# chr21 48129895
# chr22 51304566
# chrX 155270560


# train_signal_cell_A549.bigwig
# chr1 249250621
# chr2 243199373
# chr3 198022430
# chr4 191154276
# chr5 180915260
# chr6 171115067
# chr7 159138663
# chr8 146364022
# chr9 141213431
# chr10 135534747
# chr11 135006516
# chr12 133851895
# chr13 115169878
# chr14 107349540
# chr15 102531392
# chr16 90354753)
# chr17 81195210
# chr18 78077248
# chr19 59128983
# chr20 63025520
# chr21 48129895
# chr22 51304566
# chrX 155270560


# train_sites_cell_A549_tf_CTCF.bigwig
# chr2 243199373
# chr3 198022430
# chr4 191154276
# chr5 180915260
# chr6 171115067)
# chr7 159138663
# chr9 141213431
# chr10 135534747
# chr11 135006516
# chr12 133851895
# chr13 115169878
# chr14 107349540
# chr15 102531392
# chr16 90354753
# chr17 81195210
# chr18 78077248
# chr19 59128983
# chr20 63025520
# chr22 51304566
# chrX 155270560


# validate_signal_cell_HCT116.bigwig
# chr1 249250621
# chr2 243199373
# chr3 198022430
# chr4 191154276
# chr5 180915260
# chr6 171115067
# chr7 159138663
# chr8 146364022
# chr9 141213431
# chr10 135534747
# chr11 135006516
# chr12 133851895
# chr13 115169878
# chr14 107349540
# chr15 102531392
# chr16 90354753)
# chr17 81195210
# chr18 78077248
# chr19 59128983
# chr20 63025520
# chr21 48129895
# chr22 51304566
# chrX 155270560


# validate_sites_cell_HCT116_tf_CTCF.bigwig
# chr1 249250621
# chr8 146364022
# chr21 48129895