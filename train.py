import os
import torch
import math
import yaml

import pandas as pd

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import train_loop, val_loop
from model import MyDataset, CryptoTransformer, MSEPlusRank

with open('./network_config.yaml') as f:
    cfg = yaml.safe_load(f)

num_feature = cfg['num_feature']

batch_size = cfg['batch_size']
model_dim = cfg['model_dim']
ffn_dim = cfg['ffn_dim']
drop_out = cfg['drop_out']
num_heads = cfg['num_heads']
num_layers = cfg['num_layers']
max_epoch = cfg['max_epoch']
loss_lambda = cfg['loss_lambda']
lr = cfg['lr']

if cfg['experiment_name'] is not None:
    experiment_name = cfg['experiment_name']
else:
    experiment_name = f"size_{model_dim}_{ffn_dim}_nheads_{num_heads}_nlayers_{num_layers}"

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

train_length = 24236806
num_splits = 10
split_length = math.floor(train_length/num_splits)

processed_val = pd.read_csv('./processed_data/processed_val.gz')
val_dataset = MyDataset(processed_val)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

net = CryptoTransformer(model_dim=model_dim, 
                        num_heads=num_heads, 
                        num_layers=num_layers, 
                        dropout=drop_out, 
                        ffn_dim=ffn_dim, 
                        num_features=num_feature).to(device)
loss_fn = MSEPlusRank(lamda=loss_lambda)
optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

total_num = sum(p.numel() for p in net.parameters())
trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_num/1e3:.0f}K, number of trainable parameters: {trainable_num/1e3:.0f}K")

if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')
if not os.path.exists('./logs'):
    os.mkdir('./logs')
if not os.path.exists(f'./checkpoint/{experiment_name}'):
    os.mkdir(f'./checkpoint/{experiment_name}')
if not os.path.exists(f'./logs/{experiment_name}'):
    os.mkdir(f'./logs/{experiment_name}')

min_val_loss = float('inf')
best_epoch = 1
best_step = 1

tb = SummaryWriter(log_dir=f'./logs/{experiment_name}/')
for t in range(max_epoch):
    for i in range(num_splits):
        print(f"Epoch {t+1}, Train Split: {i+1}\n--------------------------")
        processed_train = pd.read_csv('./processed_data/processed_train.gz', skiprows=range(1,split_length*2), nrows=split_length)        
        train_dataset = MyDataset(processed_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loss, train_mse_loss, train_rank_loss = train_loop(train_dataloader, net, loss_fn, optimizer, device)

        val_loss, val_mse_loss, val_rank_loss = val_loop(val_dataloader, net, loss_fn, device)
        tb.add_scalar("Train Loss", train_loss, t*num_splits+i+1)
        tb.add_scalar("Train MSE Loss", train_mse_loss, t*num_splits+i+1)
        tb.add_scalar("Train Rank Loss", train_rank_loss, t*num_splits+i+1)

        tb.add_scalar("Val Loss", val_loss, t*num_splits+i+1)
        tb.add_scalar("Val MSE Loss", val_mse_loss, t*num_splits+i+1)
        tb.add_scalar("Val Rank Loss", val_rank_loss, t*num_splits+i+1)
    
        torch.save(net, f"./checkpoint/{experiment_name}/epoch_{t+1}_step_{i+1}.pt")
        if val_loss < min_val_loss:
            best_epoch = t+1
            best_step = i+1
            min_val_loss = val_loss
print(f"best epoch: {best_epoch}, best step: {best_step}, minimun validations loss: {min_val_loss:.2e}")