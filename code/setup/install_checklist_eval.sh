#!/usr/bin/env bash
# 在「环境 E（Python 3.7）」里安装 checklist==0.0.11 本体。
# 先: conda env create -f envs/environment-eval-py37.yml && conda activate checklist-eval37
set -euo pipefail

PYV=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$PYV" in
  3.6|3.7) ;;
  *) echo "[错误] 当前 Python 是 $PYV。环境 E 必须是 3.7（或 3.6），否则无法加载官方 suite pkl。" >&2; exit 1 ;;
esac

# --no-deps：不拉 transformers/torch/spacy，评测用不到
# --no-build-isolation：setup.py 的安装钩子要用到当前环境里的 notebook<7
python -m pip install "checklist==0.0.11" --no-deps --no-build-isolation

python - <<'EOF'
import sys
from checklist.test_suite import TestSuite  # noqa: F401  只检查能否 import
print("checklist 已安装，Python", sys.version.split()[0])
EOF
