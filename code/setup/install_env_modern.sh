#!/usr/bin/env bash
# 「环境 M（Python 3.10/3.11）」一键安装：跑模型、微调、工具演示、用户研究。
# 用法（在仓库根目录）：
#   python3.11 -m venv .venv && source .venv/bin/activate
#   bash code/setup/install_env_modern.sh
# GPU 用户：先按 https://pytorch.org 装好对应 CUDA 的 torch，再运行本脚本（脚本不会覆盖已装的 torch）。
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81"
# 1) 依赖（含 notebook<7，必须在 checklist 之前装好）
python -m pip install -r requirements.txt
# 2) checklist 本体：它的 setup.py 在构建时会 `pip install jupyter` 并调用 notebook.nbextensions，
#    在新环境里用默认方式安装会失败，所以用 --no-deps --no-build-isolation。
python -m pip install "checklist==0.0.11" --no-deps --no-build-isolation
# 3) pattern.en（checklist.perturb 用）首次 import 需要的 NLTK 语料
python -m nltk.downloader wordnet wordnet_ic sentiwordnet omw-1.4
# 4) Perturb.add_negation / change_names 等需要 spaCy 英文模型
python -m spacy download en_core_web_sm || echo "[提示] spaCy 模型下载失败，可稍后重试；只影响部分 Perturb 函数。"

python - <<'EOF'
import sys, torch, transformers, datasets
from checklist.editor import Editor
from checklist.perturb import Perturb
e = Editor()
print("Python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "| transformers", transformers.__version__, "| datasets", datasets.__version__)
print("Editor 模板测试:", e.template("This is {a:adj} movie.", adj=["good", "bad"]).data)
print("Perturb 测试:", Perturb.contractions("I am not sure it is good"))
EOF
echo "环境 M 安装完成。"
