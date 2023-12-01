## Introduction
This repo is about deep learning for the G-Research kaggle competition https://www.kaggle.com/competitions/g-research-crypto-forecasting/overview. 

Since we do not have access to the original test set, we split the 
first 40% of supplemental_train.csv as validation set and last 40% of as test set. We use techniques of purging 
and embargo to prevent leakage of feature information.
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
Use `train.ipynb` and `evaluation.ipynb` to train and evaluate the model. Run the following command to open the tensorboard and monitor the loss during the training process
```
tensorboard --logdir ./logs
```