#!/usr/bin/env bash
# Assemble the submission zip. Run from student_resource/ AFTER outputs exist and
# validate_submission.py has printed PASS.
set -euo pipefail
cd "$(dirname "$0")"

test -s output/matching_results.tsv || { echo "missing output/matching_results.tsv"; exit 1; }
test -s output/candidate_pairs.tsv  || { echo "missing output/candidate_pairs.tsv"; exit 1; }

python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test | tail -3

ZIP=submission_$(date +%Y%m%d_%H%M).zip
rm -f "$ZIP"
# .venv and the cached parquet are build artefacts, not source; the model json is
# small and IS shipped so the run is reproducible.
zip -qr "$ZIP" \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    Documentation_template.md \
    code/business_entity_resolution \
    -x '*/.venv/*' '*/__pycache__/*' '*.parquet' '*/model_dryrun/*' \
       '*/model_tune_sample/*' '*/model_v2_test/*' '*/model_v3_test/*'
echo "built $ZIP"
du -h "$ZIP"
unzip -l "$ZIP" | tail -3
