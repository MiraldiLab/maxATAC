# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq

## Install Instructions

### 1. To install maxATAC begin by downloading the data reopository located on github: maxATAC_data

 `git clone https://github.com/MiraldiLab/maxATAC_data.git`

### 2. To ensure that the data is located in a central location copy maxATAC_data to local location:

```bash
mkdir -p /opt/maxatac/data/
cp -r ./maxATAC_data /opt/maxatac/
```

### 3. Then Download hg38.2bit file from UCSC in a location of your choice

```bash
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit
```

### 4. Installing maxATAC

#### Installing with Conda

1. Create a conda environment for maxATAC with `conda create -n maxatac python=3.9 samtools wget bedtools ucsc-bedgraphtobigwig pigz`

> If you get an error installing ucsc-bedgraphtobigwig try `conda install -c bioconda ucsc-bedgraphtobigwig`

2. Install maxATAC with `pip install maxatac`

3. Test installation with `maxatac -h`

4. Download reference data with `maxatac data`

#### Installing with python virtualenv

1. Create a virtual environment for maxATAC with `virtualenv -p python3.9 maxatac`.

2. Install required packages and make sure they are on your PATH: samtools, bedtools, bedGraphToBigWig, wget, git, pigz.

3. Install maxatac with `pip install maxatac`

4. Test installation with `maxatac -h`

#### Running maxATAC with a docker

A docker image of maxATAC can be found on our lab [dockerpage](https://hub.docker.com/repository/docker/miraldi/maxatac)

1. install and run docker
2. Run `docker pull miraldi/maxatac:v0.0.3` to pull the docker image onto your device
3. Run maxATAC through docker `docker run --rm -ti miraldi/maxatac:v0.0.3 /bin/bash`

