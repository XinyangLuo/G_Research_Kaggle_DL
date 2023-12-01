import os
import pandas as pd

from utils import process_time_series_feature

if not os.path.exists('./processed_data'):
    os.mkdir('./processed_data')

path = './g-research-crypto-forecasting/'
asset_list = pd.read_csv(path+"asset_details.csv")
train_dataset = pd.read_csv(path+"train.csv")
validation_dataset = pd.read_csv(path+"supplemental_train.csv")

# purging to avoid leakage of information
train_dataset = train_dataset.iloc[:round(len(train_dataset)*0.95)]

val_len = len(validation_dataset)
test_dataset = validation_dataset.iloc[-round(0.45*val_len):]
validation_dataset = validation_dataset.iloc[:round(0.45*val_len)]

test_val_gap = (test_dataset['timestamp'].min() - validation_dataset['timestamp'].max())/3600/24
train_val_gap = (validation_dataset['timestamp'].min() - train_dataset['timestamp'].max())/3600/24
print(f"gap between train dataset and validation dataset: {train_val_gap:.2f} days")
print(f"gap between validation dataset and test_dataset: {test_val_gap:.2f} days")

train_dataset.dropna(axis=0, subset="Target", inplace=True)
validation_dataset.dropna(axis=0, subset="Target", inplace=True)
test_dataset.dropna(axis=0, subset="Target", inplace=True)

train_dataset.reset_index(inplace=True)
validation_dataset.reset_index(inplace=True)
test_dataset.reset_index(inplace=True)

print("------------------------------------\nprocess training data\n------------------------------------")
processed_train = process_time_series_feature(train_dataset, verbose=True)
processed_train.to_csv('./processed_data/processed_train.gz', compression='gzip')
del train_dataset, processed_train

print("------------------------------------\nprocess validation data\n------------------------------------")
processed_val = process_time_series_feature(validation_dataset, verbose=True)
processed_val.to_csv('./processed_data/processed_val.gz', compression='gzip')
del validation_dataset, processed_val

print("------------------------------------\nprocess test data\n------------------------------------")
processed_test = process_time_series_feature(test_dataset, verbose=True)
processed_test.to_csv('./processed_data/processed_test.gz', compression='gzip')
del test_dataset, processed_test