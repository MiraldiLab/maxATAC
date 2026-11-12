#################################################################
# quant-maxATAC (CPU)
#
# Build:  docker build -t quant-maxatac --build-arg MAXATAC_VERSION=2.0.0b2 .
#         MAXATAC_VERSION is the release without its "v" (the git tag is not copied into the image). It decides
#         which data tag `maxatac data` installs; without it the image is a development version and installs the
#         latest data.
#
# The image holds no reference data or models (the hg38.2bit alone is ~800 MB). Keep them on the host under
# ~/opt/maxatac and mount that directory; `maxatac data` fills it once and later runs reuse it:
#
#   docker run --rm --user "$(id -u):$(id -g)" -v ~/opt/maxatac:/home/maxatac/opt/maxatac \
#       quant-maxatac maxatac data
#   docker run --rm --user "$(id -u):$(id -g)" -v ~/opt/maxatac:/home/maxatac/opt/maxatac -v "$PWD":/work \
#       quant-maxatac maxatac predict --tf_name CTCF --signal sample.bw --prefix sample_CTCF
#
# Singularity/Apptainer mounts your home directory by default, so ~/opt/maxatac/data on the host is used as is:
#   apptainer exec docker://<image> maxatac predict ...
#################################################################

ARG PYTHON_VERSION=3.11

########## Build stage: compile the Python packages into a virtualenv ##########
FROM python:${PYTHON_VERSION}-slim-bookworm AS build

# Compilers and headers for packages without wheels (ushuffle, py2bit, ...)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git zlib1g-dev libbz2-dev liblzma-dev \
        libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# setuptools 82 removed pkg_resources, which older dependencies (e.g. TF 2.15-era tensorboard) still import
RUN pip install --no-cache-dir --upgrade pip "setuptools<82" wheel

# The PyPI release of ushuffle does not build: install it from source first, as in .github/workflows/ci.yml.
# --no-build-isolation needs the build tools above. If the build fails, retry with Cython < 3
ARG USHUFFLE_SOURCE=git+https://github.com/guma44/ushuffle.git@b54df34f60246ff236ec58bbccb40af3cea608ed
RUN (pip install --no-cache-dir --no-build-isolation "${USHUFFLE_SOURCE}" || \
     (pip install --no-cache-dir "cython<3" && pip install --no-cache-dir --no-build-isolation "${USHUFFLE_SOURCE}")) && \
    python -c "from ushuffle import shuffle; shuffle(b'ACGTACGTAC', 2)"

COPY . /src

# Without .git, setup.py reads the version from maxatac/git_version
ARG MAXATAC_VERSION=""
RUN if [ -n "${MAXATAC_VERSION}" ]; then printf "%s" "${MAXATAC_VERSION#v}" > /src/maxatac/git_version; fi && \
    pip install --no-cache-dir /src

########## Runtime stage ##########
FROM python:${PYTHON_VERSION}-slim-bookworm

ARG MAXATAC_VERSION=""
# GHCR links the image to this repository (release.yml passes the repository that built it)
ARG SOURCE_URL=https://github.com/MiraldiLab/quant-maxATAC
LABEL org.opencontainers.image.title="quant-maxATAC" \
      org.opencontainers.image.description="Deep neural networks predicting quantitative TF ChIP-seq signal from ATAC-seq signal and DNA sequence" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${MAXATAC_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0"

# git, wget: `maxatac data`. samtools, bedtools, pigz, bedGraphToBigWig: `maxatac prepare`; bedtools also for
# pybedtools in predict/benchmark
ARG BEDGRAPHTOBIGWIG_URL=https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bedGraphToBigWig
RUN apt-get update && \
    apt-get install -y --no-install-recommends bedtools samtools pigz git wget ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q -O /usr/local/bin/bedGraphToBigWig "${BEDGRAPHTOBIGWIG_URL}" && \
    chmod +x /usr/local/bin/bedGraphToBigWig && \
    (bedGraphToBigWig 2>&1 || true) | grep -q "usage"

COPY --from=build /opt/venv /opt/venv

# A fixed HOME, so the data directory (~/opt/maxatac/data) is the same for any --user. World-writable because
# --user runs as an arbitrary uid
ENV PATH=/opt/venv/bin:$PATH \
    HOME=/home/maxatac \
    MPLCONFIGDIR=/tmp/matplotlib \
    TF_CPP_MIN_LOG_LEVEL=2
RUN mkdir -p /home/maxatac/opt/maxatac /work && chmod -R 777 /home/maxatac /work

WORKDIR /work

RUN maxatac -h > /dev/null

CMD ["maxatac", "-h"]
