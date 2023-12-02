import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)

class CryptoTransformer(nn.Module):
    def __init__(self, model_dim=64, num_heads=2, num_layers=2, dropout=0.2, ffn_dim=128, num_features=40):
        super().__init__()
        
        self.input_up = nn.Linear(num_features, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = model_dim, 
                                                        nhead = num_heads,
                                                        dim_feedforward = ffn_dim,
                                                        dropout = dropout,
                                                        batch_first = True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(model_dim, 1)
        self.apply(init_weights)
        
    def forward(self, x, mask):
        emb_x = self.input_up(x)
        enc_x = self.encoder(emb_x, src_key_padding_mask=~mask)
        output = self.output(enc_x)
        return output
    
class MyDataset(Dataset):
    def __init__(self, df):
        self.groups = df.groupby(['timestamp'])
        self.timestamps = df['timestamp'].unique()
        self.feature_cols = ['Count', 'Open', 'High', 'Low', 'Close','Volume', 'VWAP', 'Vol', 
                             'upper_shadow', 'lower_shadow', 'upper_ratio', 'lower_ratio', 
                             'Average_volume', 'Regret', 'MA5', 'MA15', 'MA30', 'MA60', 'VMA5', 
                             'VMA15', 'VMA30', 'VMA60', 'RV5', 'RV15', 'RV30', 'RV60', 'ADX', 'ADXR',
                             'APO', 'AROONOSC', 'BOP', 'CCI', 'DX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM',
                             'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'RSI', 'TRIX', 'ULTOSC', 'WILLR',
                             'AD', 'ADOSC', 'OBC', 'ATR', 'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 
                             'Binance Coin',  'Bitcoin', 'Bitcoin Cash', 'Cardano', 'Dogecoin', 'EOS.IO', 'Ethereum', 
                             'Ethereum Classic', 'IOTA', 'Litecoin', 'Maker', 'Monero', 'Stellar', 'TRON']
    
    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        ts = self.timestamps[idx]
        data = self.groups.get_group(ts)
        x = data.loc[:, self.feature_cols].values
        num_assets = x.shape[0]
        if num_assets < 14:
            padding = np.zeros((14-num_assets, x.shape[1]))
            x = np.concatenate((x, padding), axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        
        y = data.loc[:, 'Target'].values
        y = torch.tensor(y, dtype=torch.float32)
        y = F.pad(y, (0, 14-num_assets))
        return x, y, num_assets

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, mask):
        return (mask*(y_pred-y_true)**2).mean()

class PairWiseRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, mask):
        batch_size, n = y_pred.shape
        y_pred_diff = y_pred.reshape(batch_size, n, 1) - y_pred.reshape(batch_size, 1, n)
        y_true_diff = y_true.reshape(batch_size, n, 1) - y_true.reshape(batch_size, 1, n)
        mask = mask.reshape(batch_size, 1, n)*mask.reshape(batch_size, n, 1)
        total_loss = F.relu(-y_pred_diff*y_true_diff*mask)
        return total_loss.mean()

class MSEPlusRank(nn.Module):
    def __init__(self, lamda=1):
        super().__init__()
        self.lamda = lamda
        self.mse = MaskedMSELoss()
        self.rankloss = PairWiseRankingLoss()

    def forward(self, y_pred, y_true, mask):
        loss1 = self.mse(y_pred, y_true, mask)
        loss2 = self.rankloss(y_pred, y_true, mask)
        coef = (loss2/loss1).detach()
        loss = coef*loss1 + self.lamda*loss2
        return loss, loss1, loss2