# VisBERT: Hidden-State Visualizations for Transformers

This demo visualizes the findings from our paper [How Does BERT Answer Questions? A Layer-Wise Analysis of Transformer Representations](https://arxiv.org/abs/1909.04925).
The tool consists of a Python (>=3.6.1) Flask app and does currently rely on three fine-tuned Pytorch BERT models.

## Getting Started
Install requirements with pip:

`pip install -r docker/requirements.txt`

Install NLTK Punkt:

`python -m nltk.downloader punkt`

Run app with model directory as argument. The model directory must currently contain the models: 'squad.bin', 'hotpot_distract.bin' and 'babi.bin', which are available [here](https://drive.google.com/drive/folders/1RZvlZmhxiJKiAagwvE5vjmevmiqqjR0r):

`python src/app.py {model_directory}`


## Cite
When using our tool, please cite the following paper:
```
@article{van_Aken_2020,
   title={VisBERT: Hidden-State Visualizations for Transformers},
   author={van Aken, Betty and Winter, Benjamin and Löser, Alexander and Gers, Felix A.},
   journal={WWW '20: Companion Proceedings of the Web Conference 2020},
   publisher={ACM Press},
   year={2020}
}
```
