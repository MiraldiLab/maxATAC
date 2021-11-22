# Peaks

The `peaks` will take a maxATAC prediction `.bw` signal track and call bins that are above a specific threshold. These bins will be merged and output as BED intervals that can be visualized and used for downstream analysis. 

## Example

```maxatac peaks -i GM12878_CTCF.bw -o ./peaks -bin 32 -threshold .75```

## Required Arguments
