# VisBERT: Hidden-State Visualizations for BERT

This demo visualizes the findings from our paper [How Does BERT Answer Questions? A Layer-Wise Analysis of Transformer Representations](https://arxiv.org/abs/1909.04925).
The tool consists of a Python (>3.6) Flask app and does currently rely on three fine-tuned Pytorch BERT models.

## Get Started
Install requirements with pip:

`pip install -r docker/requirements.txt`

Install NLTK Punkt:

`python -m nltk.downloader punkt`

Run app with model directory as argument (the model directory must currently contain the models: 'squad.bin', 'hotpot_distract.bin' and 'babi.bin'):

`python src/app.py {model_directory}`


## Cite
When using our tool, please cite the following paper:
```
@article{van_Aken_2019,
   title={How Does BERT Answer Questions?},
   journal={Proceedings of the 28th ACM International Conference on Information and Knowledge Management  - CIKM  ’19},
   publisher={ACM Press},
   author={van Aken, Betty and Winter, Benjamin and Löser, Alexander and Gers, Felix A.},
   year={2019}
}
```
