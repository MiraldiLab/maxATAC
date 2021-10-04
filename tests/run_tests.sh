#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pip3 install -r $DIR/../test_requirements.txt --use-feature=2020-resolver

cd $DIR
mkdir -p data
cd data

if [ ! -f "hg19.2bit" ] ; then
    echo "Downloading hg19.2bit"
    wget -q --show-progress ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.2bit -O hg19.2bit
fi

if [ ! -f "predict_signal_cell_GM12878.bigwig" ] ; then
    echo "Downloading predict_signal_cell_GM12878.bigwig"
    wget -q --show-progress https://guanfiles.dcmb.med.umich.edu/dnase_bigwig/GM12878.bigwig -O predict_signal_cell_GM12878.bigwig
fi

if [ ! -f "train_signal_cell_A549.bigwig" ] ; then
    echo "Downloading train_signal_cell_A549.bigwig"
    wget -q --show-progress https://guanfiles.dcmb.med.umich.edu/dnase_bigwig/A549.bigwig -O train_signal_cell_A549.bigwig
fi

if [ ! -f "validate_signal_cell_HCT116.bigwig" ] ; then
    echo "Downloading validate_signal_cell_HCT116.bigwig"
    wget -q --show-progress https://guanfiles.dcmb.med.umich.edu/dnase_bigwig/HCT116.bigwig -O validate_signal_cell_HCT116.bigwig
fi

if [ ! -f "average.bigwig" ] ; then
    echo "Downloading average.bigwig"
    wget -q --show-progress https://guanfiles.dcmb.med.umich.edu/dnase_bigwig/avg.bigwig -O average.bigwig
fi

if [ ! -f "train_sites_cell_A549_tf_CTCF.bigwig" ] ; then
    echo "Downloading train_sites_cell_A549_tf_CTCF.bigwig"
    wget -q --show-progress https://guanfiles.dcmb.med.umich.edu/chipseq_conservative_refine_bigwig/CTCF_A549.bigwig -O train_sites_cell_A549_tf_CTCF.bigwig
fi

if [ ! -f "validate_sites_cell_HCT116_tf_CTCF.bigwig" ] ; then
    echo "Downloading validate_sites_cell_HCT116_tf_CTCF.bigwig"
    wget -q --show-progress https://guanfiles.dcmb.med.umich.edu/test_chipseq_conservative_refine_bigwig/CTCF_HCT116.bigwig -O validate_sites_cell_HCT116_tf_CTCF.bigwig
fi

if [ ! -f "GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed" ] ; then
    echo "Downloading GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed"
    wget -q --show-progress https://ftp.ncbi.nlm.nih.gov/geo/series/GSE143nnn/GSE143104/suppl/GSE143104_ENCFF551HTC_pseudoreplicated_IDR_thresholded_peaks_hg19.bigBed
fi

cd ..

pytest -v --cov=maxatac --forked .