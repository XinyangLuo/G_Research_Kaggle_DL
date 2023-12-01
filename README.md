## Introduction
This repo is about deep learning for the G-Research kaggle competition https://www.kaggle.com/competitions/g-research-crypto-forecasting/overview.
## Install the dependency
Run the following scripts to install the dependency, if you want to use pytorch with cuda, you should install the correct version by yourself
```
pip install -r requirements.txt
```

## Preprocess data
First, download the data from the competition website.

Then, run the following scripts to process data, this may take a while since the data size is large
```
python process_data.py
```

## Training and Evaluation
Use `train.ipynb` and `evaluation.ipynb` to train and evaluate the model