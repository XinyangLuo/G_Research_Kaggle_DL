## Introduction
This repo is about deep learning for the G-Research kaggle competition https://www.kaggle.com/competitions/g-research-crypto-forecasting/overview. 

Since we do not have access to the original test set, we split the 
first 40% of supplemental_train.csv as validation set and last 40% of as test set. We use techniques of purging 
and embargo to prevent leakage of feature information.

## Install the dependency
Run the following scripts to install the dependency, if you want to use pytorch with cuda, you should install the
suitable version by yourself
```
pip install -r requirements.txt
```

## Preprocess data
First, download the data from the competition website.

Then, run the following scripts to process data, this may take a while since the data size is large
```
python process_data.py
```

## Training
Modify the hyperparameter of the network in `network_config.yaml`, then run the following command to train 
you model. The trained model will be saved in the *./checkpoint/{$experiment_name}* path.
```
python train.py
```
Run the following command to open the tensorboard and monitor the loss during the training process
```
tensorboard --logdir ./logs
```

## Evaluation
Use `evaluation.ipynb` to calculate the unweighted and weighted Pearson correlation. Modify the weight path
to load the demanding model.