# quant-maxATAC: deep neural network models for quantitative transcription factor binding prediction from ATAC-seq

<!-- TODO: `pip install maxatac` installs the binary maxATAC v1 release from PyPI. Replace the git install commands
below with the PyPI package name/version once quant-maxATAC is published. The docker image below is also v1. -->

## Install Instructions

The reference data and models are installed with `maxatac data` once quant-maxATAC is installed (step 4). To install them by hand instead, follow steps 1-3.

### 1. Download the data repository located on github: [quant-maxATAC_data](https://github.com/MiraldiLab/quant-maxATAC_data)

This replaces the maxATAC v1 [maxATAC_data](https://github.com/MiraldiLab/maxATAC_data) repository, whose binary models quant-maxATAC does not use. If you have an existing `~/opt/maxatac/data`, move it out of the way first.

### 2. Clone it into the location quant-maxATAC looks in by default:

```bash
mkdir -p ~/opt/maxatac
git clone --depth 1 https://github.com/MiraldiLab/quant-maxATAC_data.git ~/opt/maxatac/data
```

To match a quant-maxATAC release, clone the data tag of the same name, e.g. `git clone --depth 1 --branch v2.0.0b1 ...`. `maxatac data` does this for you.

### 3. Then download the hg38.2bit file from UCSC next to the hg38 reference files

```bash
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit -O ~/opt/maxatac/data/hg38/hg38.2bit
```

### 4. Installing maxATAC

#### Installing with Conda

1. Create a conda environment for maxATAC with `conda create -n maxatac python=3.9 samtools wget bedtools ucsc-bedgraphtobigwig pigz`

> If you get an error installing ucsc-bedgraphtobigwig try `conda install -c bioconda ucsc-bedgraphtobigwig`

2. Install quant-maxATAC with `pip install git+https://github.com/MiraldiLab/quant-maxATAC.git@quant`

3. Test installation with `maxatac -h`

4. Download reference data with `maxatac data`

#### Installing with python virtualenv

1. Create a virtual environment for maxATAC with `virtualenv -p python3.9 maxatac`.

2. Install required packages and make sure they are on your PATH: samtools, bedtools, bedGraphToBigWig, wget, git, pigz.

3. Install quant-maxATAC with `pip install git+https://github.com/MiraldiLab/quant-maxATAC.git@quant`

4. Test installation with `maxatac -h`

#### Running maxATAC with a docker

A docker image of maxATAC can be found on our lab [dockerpage](https://hub.docker.com/repository/docker/miraldi/maxatac)

1. install and run docker
2. Run `docker pull miraldi/maxatac:v0.0.3` to pull the docker image onto your device
3. Run maxATAC through docker `docker run --rm -ti miraldi/maxatac:v0.0.3 /bin/bash`

