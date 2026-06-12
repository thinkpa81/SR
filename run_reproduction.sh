set -euo pipefail
cd "$(dirname "$0")"
python python/00_run_full_pipeline.py
python python/15_validate_submission_consistency.py
