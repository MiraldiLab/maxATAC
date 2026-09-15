# Data

The `data` function downloads the reference data and models that the other functions look for by default under `~/opt/maxatac/data`:

* the [maxATAC_data](https://github.com/MiraldiLab/maxATAC_data) repository (blacklists, chromosome sizes, TF models with their `*_cross_celltype.tsv` threshold calibration tables, and the `prepare` shell scripts), cloned into `~/opt/maxatac/data`
* the UCSC `.2bit` genome sequence for each requested genome, downloaded into `~/opt/maxatac/data/<genome>/`

<!-- TODO: `maxatac data` currently clones MiraldiLab/maxATAC_data, which holds the binary maxATAC v1 models. Update
this section (and maxatac/analyses/data.py) once the quant-maxATAC models and calibration tables have a home. -->

`git` and `wget` must be on your PATH.

## Example

```bash
maxatac data
maxatac data --genome hg38 hg19
```

## Optional Arguments

### `--genome`

One or more genome builds to download the `.2bit` sequence for, or `all` for `hg38 hg19 mm10`. Only hg38 has been tested extensively. Default: `hg38`

### `-o`, `--output`

Base directory to install into; data is placed in `<output>/maxatac/data`. Default: `~/opt`

### `--loglevel`

Logging level (`fatal`, `error`, `warning`, `info`, `debug`). Default: `info`.
