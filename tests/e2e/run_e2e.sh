#!/usr/bin/env bash
# End-to-end test of the user path: maxatac data -> maxatac predict -> maxatac benchmark
#
# Predicts one TF on a slice of chr1 from the GM12878 test inputs in
# https://github.com/XiaotingChen/quant-maxATAC_test_data (Git LFS: clone with `git lfs` installed)
# and benchmarks the prediction against the ChIP-seq gold standard on the same slice.
#
# Used by .github/workflows/e2e.yml; runs the same way on a workstation or the cluster:
#   TEST_DATA=/path/to/quant-maxATAC_test_data/data bash tests/e2e/run_e2e.sh
#
# Environment (all optional):
#   TEST_DATA   directory with the test inputs              Default: test_data/data
#   OUT         output directory                            Default: e2e_out
#   TF          TF to predict (model from `maxatac data`)   Default: CTCF
#   ROI_START   start of the chr1 slice                     Default: 1000000
#   ROI_END     end of the chr1 slice                       Default: 11000000
#   DATA_VERSION  data repository tag/branch for `maxatac data --data_version`  Default: matches the code version
#   SKIP_DATA   1 = reuse the installed data, skip `maxatac data`
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEST_DATA="${TEST_DATA:-test_data/data}"
OUT="${OUT:-e2e_out}"
TF="${TF:-CTCF}"
CHROM=chr1
ROI_START="${ROI_START:-1000000}"
ROI_END="${ROI_END:-11000000}"
GENOME=hg38
DATA_DIR="${HOME}/opt/maxatac/data"
PREFIX="GM12878_${TF}_e2e"

ATAC_SIGNAL="${TEST_DATA}/GM12878_RP20M_minmax_percentile99_chr1.bw"
GOLD_STANDARD="${TEST_DATA}/GM12878__${TF}.bw"
QUANT_GOLD_STANDARD="${TEST_DATA}/GM12878__${TF}_quantLinearFC_chr1.bw"

CHECK="python ${HERE}/check_outputs.py"

mkdir -p "${OUT}"

for input in "${ATAC_SIGNAL}" "${GOLD_STANDARD}" "${QUANT_GOLD_STANDARD}"; do
    # An LFS pointer is ~130 bytes: catch a clone made without git-lfs early
    if [ ! -s "${input}" ] || [ "$(wc -c < "${input}")" -lt 1024 ]; then
        echo "::error::Missing test input or unresolved Git LFS pointer: ${input}"
        exit 1
    fi
done

echo "::group::maxatac data"
if [ "${SKIP_DATA:-0}" != "1" ]; then
    maxatac data --genome "${GENOME}" --yes ${DATA_VERSION:+--data_version "${DATA_VERSION}"}
fi
${CHECK} data --data_dir "${DATA_DIR}" --genome "${GENOME}" --tf "${TF}"
echo "::endgroup::"

echo "::group::Region of interest"
# Prediction is limited to one slice of chr1 so it runs on a CPU runner in minutes
printf "%s\t%s\t%s\n" "${CHROM}" "${ROI_START}" "${ROI_END}" > "${OUT}/roi.bed"
# chr1-only chrom sizes, so the sliding windows are not built genome wide
awk -v c="${CHROM}" '$1 == c' "${DATA_DIR}/${GENOME}/${GENOME}.chrom.sizes" > "${OUT}/${CHROM}.chrom.sizes"
# benchmark has no region option: restrict it to the ROI with a whitelist bigWig
${CHECK} whitelist --roi "${OUT}/roi.bed" --chrom_sizes "${OUT}/${CHROM}.chrom.sizes" --out "${OUT}/roi_whitelist.bw"
cat "${OUT}/roi.bed"
echo "::endgroup::"

echo "::group::maxatac predict"
maxatac predict \
    --tf_name "${TF}" \
    --signal "${ATAC_SIGNAL}" \
    --chromosomes "${CHROM}" \
    --chrom_sizes "${OUT}/${CHROM}.chrom.sizes" \
    --roi "${OUT}/roi.bed" \
    --threads 1 \
    --batch_size 2000 \
    --prefix "${PREFIX}" \
    --output "${OUT}/predict"
${CHECK} prediction --prediction "${OUT}/predict/${PREFIX}.bw" --peaks "${OUT}/predict/${PREFIX}_peaks.bed" \
    --roi "${OUT}/roi.bed"
echo "::endgroup::"

echo "::group::maxatac benchmark"
maxatac benchmark \
    --bw "${OUT}/predict/${PREFIX}.bw" \
    --gold_standard "${GOLD_STANDARD}" \
    --whitelist_bw "${OUT}/roi_whitelist.bw" \
    --chromosomes "${CHROM}" \
    --bin_size 200 \
    --agg max \
    --plot \
    --prefix "${PREFIX}" \
    --output_directory "${OUT}/benchmark"
echo "::endgroup::"

echo "::group::Metrics"
${CHECK} report \
    --prc "${OUT}/benchmark/${PREFIX}_${CHROM}_200bp_PRC.tsv" \
    --prediction "${OUT}/predict/${PREFIX}.bw" \
    --quant_gold_standard "${QUANT_GOLD_STANDARD}" \
    --blacklist_bw "${DATA_DIR}/${GENOME}/${GENOME}_maxatac_blacklist.bw" \
    --roi "${OUT}/roi.bed" \
    --baseline "${HERE}/baseline.json" \
    --metrics_out "${OUT}/metrics.json"
echo "::endgroup::"
