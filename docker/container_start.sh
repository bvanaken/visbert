#!/bin/sh

git clone https://github.com/bvanaken/visbert.git /visbert

python3 -m nltk.downloader punkt

python3 /visbert/src/app.py /models_dir/