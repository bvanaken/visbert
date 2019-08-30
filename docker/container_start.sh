#!/bin/sh

git clone https://github.com/bvanaken/visbert.git /visbert

export PIP_DOWNLOAD_CACHE=/models_dir/cache/

pip install torch==1.1.0 sklearn nltk pytorch-transformers==1.1.0
python3 -m nltk.downloader punkt

python3 /visbert/src/app.py /models_dir/