# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq

## Install Instructions

### 1. To install maxATAC begin by downloading the data reopository located on github: maxATAC_data
 git clone https://github.com/MiraldiLab/maxATAC_data.git

### 2. To ensure that the data is located in a central location copy maxATAC_data to local location:

```
mkdir -p /opt/maxatac/data/
cp -r ./maxATAC_data /opt/maxatac/
```

### 3. Then Download hg38.2bit file from UCSC in a location of your choice

```
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit
```

### 4. Install maxATAC

To install maxATAC we provide a few options to our users.

#### a) using pip

##### I) Begin first by creating your environment:

```
python3.9 -m venv env_name
```
or by using conda 
```
conda create -n my_env python=3.9 or python3.9 -m venv env_name
```

##### II) activate your environment

##### III) 

#### b) using conda


#### c) using Docker




