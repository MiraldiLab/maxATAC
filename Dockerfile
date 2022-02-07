#################################################################
# Dockerfile
#
# Software:         maxATAC
# Software Version: v0.0.3
# Description:      python based suite for transcription factor binding prediction
# Website:          https://github.com/MiraldiLab/maxATAC
# Website2:         https://github.com/MiraldiLab/maxATAC_data
# Provides:         Transcription Factor Binding Prediction
# Base Image:       biowardrobe2/samtools:v1.4/bedtools/ucsc-bedgraphtobigwig/pigz
# Build Cmd:        docker build --rm -t maxatac/maxatac:v0.0.1 -f Dockerfile .
# Pull Cmd:         docker pull maxatac/maxatac:v0.0.1
# Run Cmd:          docker run --rm -ti maxatac/maxatac:v0.0.1
#################################################################


# Dockerfile, Image, Container

# docker build --rm -t maxatac/maxatac:v0.0.1 -f Dockerfile .

FROM python:3.9

WORKDIR /tmp

ENV VERSION_BEDTOOLS 2.26.0
ENV URL_BEDTOOLS "https://github.com/arq5x/bedtools2/releases/download/v${VERSION_BEDTOOLS}/bedtools-${VERSION_BEDTOOLS}.tar.gz"

ENV VERSION_HTSLIB 1.12
ENV URL_HTSLIB "https://github.com/samtools/htslib/releases/download/${VERSION_HTSLIB}/htslib-${VERSION_HTSLIB}.tar.bz2"

ENV VERSION_SAMTOOLS 1.12
ENV URL_SAMTOOLS "https://github.com/samtools/samtools/releases/download/${VERSION_SAMTOOLS}/samtools-${VERSION_SAMTOOLS}.tar.bz2"

ENV URL_UCSC "http://hgdownload.soe.ucsc.edu/admin/exe/userApps.src.tgz"


RUN apt-get update && \
    apt-get install -y gcc-10-base libgcc-10-dev libxml2-dev libcurl4-openssl-dev libssl-dev pandoc libgtextutils-dev  libmariadb-dev-compat libmariadb-dev libpng-dev uuid-dev libncurses5-dev libbz2-dev liblzma-dev pigz rsync && \
### Install htslib
    wget -q -O - $URL_HTSLIB | tar -jxv && \
    cd htslib-${VERSION_HTSLIB} && \
    ./configure --prefix=/usr/local/ && \
    make -j 4 && \
    make install && \
    cd .. && \
### Install samtools
    wget -q -O - $URL_SAMTOOLS | tar -jxv && \
    cd samtools-${VERSION_SAMTOOLS} && \
    ./configure --prefix=/usr/local/ && \
    make -j 4 && \
    make install && \
    cd .. && \
### Install bedtools
    wget -q -O - $URL_BEDTOOLS | tar -zxv && \
    cd bedtools2 && \
    make -j 4 && \
    cd .. && \
    cp ./bedtools2/bin/bedtools /usr/local/bin/ && \
### Install UCSC User Apps
    wget -q -O - $URL_UCSC | tar -zxv --strip-components=2 && \
    make -j 2 && \
    cp ./bin/* /usr/local/bin/ &&\
    rm -rf ./* &&\
    #pip install maxatac==0.0.3
    strip /usr/local/bin/*; true



#COPY . .
#
#RUN pip install maxatac==0.0.3
#git clone https://github.com/MiraldiLab/maxATAC_data.git \
#
##COPY . .
#
#CMD ["maxatac"]
