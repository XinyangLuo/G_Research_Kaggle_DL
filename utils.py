import torch
import numpy as np
import pandas as pd
import talib as tb

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

asset_names = ['Binance Coin', 'Bitcoin', 'Bitcoin Cash', 'Cardano',
               'Dogecoin', 'EOS.IO', 'Ethereum', 'Ethereum Classic',
               'IOTA', 'Litecoin', 'Maker', 'Monero', 'Stellar', 'TRON']

def process_one_asset(asset):
    asset.sort_values(by='timestamp', inplace=True)
    scaler = StandardScaler()
    
    asset.loc[np.isinf(asset['VWAP']), 'VWAP'] = asset[~np.isinf(asset['VWAP'])]['VWAP'].mean()
    asset.loc[np.isnan(asset['VWAP']), 'VWAP'] = asset[~np.isnan(asset['VWAP'])]['VWAP'].mean()

    asset['Vol'] = (asset['Close'].diff()/asset['Close'])**2
    # add some time series feature

    open_price = asset['Open']
    close_price = asset['Close']
    low_price = asset['Low']
    high_price = asset['High']
    volume = asset['Volume']
    vwap = asset['VWAP']

    asset['upper_shadow'] = asset['High'] - asset[['Open', 'Close']].max(axis=1)
    asset['lower_shadow'] = asset[['Open', 'Close']].min(axis=1) - asset['Low']
    asset['upper_ratio'] = asset['upper_shadow']/(asset['High'] - asset['Low'])
    asset['lower_ratio'] = asset['lower_shadow']/(asset['High'] - asset['Low'])
    asset['Average_volume'] = asset['Volume']/asset['Count']
    asset['Regret'] = asset['VWAP']/asset['Close']

    asset['MA5'] = tb.EMA(close_price, 5)
    asset['MA15'] = tb.EMA(close_price, 15)
    asset['MA30'] = tb.EMA(close_price, 30)
    asset['MA60'] = tb.EMA(close_price, 60)

    asset['VMA5'] = tb.EMA(volume, 5)
    asset['VMA15'] = tb.EMA(volume, 15)
    asset['VMA30'] = tb.EMA(volume, 30)
    asset['VMA60'] = tb.EMA(volume, 60)

    asset['RV5'] = asset['Vol'].rolling(5).sum()
    asset['RV15'] = asset['Vol'].rolling(15).sum()
    asset['RV30'] = asset['Vol'].rolling(30).sum()
    asset['RV60'] = asset['Vol'].rolling(60).sum()

    asset['ADX'] = tb.ADX(high_price, low_price, close_price, timeperiod=28)
    asset['ADXR'] = tb.ADXR(high_price, low_price, close_price, timeperiod=28)
    asset['APO'] = tb.APO(close_price, fastperiod=24, slowperiod=52)
    asset['AROONOSC'] = tb.AROONOSC(high_price, low_price, timeperiod=14)
    asset['BOP'] = tb.BOP(open_price, high_price, low_price, close_price)
    asset['CCI'] = tb.CCI(high_price, low_price, close_price, timeperiod=14)
    asset['DX'] = tb.DX(high_price, low_price, close_price, timeperiod=14)
    asset['MFI'] = tb.MFI(high_price, low_price, close_price, volume, timeperiod=14)
    asset['MINUS_DI'] = tb.MINUS_DI(high_price, low_price, close_price, timeperiod=14)
    asset['MINUS_DM'] = tb.MINUS_DM(high_price, low_price, timeperiod=14)
    asset['MOM'] = tb.MOM(close_price, timeperiod=10)
    asset['PLUS_DI'] = tb.PLUS_DI(high_price, low_price, close_price, timeperiod=14)
    asset['PLUS_DM'] = tb.PLUS_DM(high_price, low_price, timeperiod=14)
    asset['PPO'] = tb.PPO(close_price, fastperiod=12, slowperiod=26, matype=0)
    asset['ROC'] = tb.ROC(close_price, timeperiod=10)
    asset['ROCP'] = tb.ROCP(close_price, timeperiod=10)
    asset['RSI'] = tb.RSI(close_price, timeperiod=14)
    asset['TRIX'] = tb.TRIX(close_price, timeperiod=30)
    asset['ULTOSC'] = tb.ULTOSC(high_price, low_price, close_price, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    asset['WILLR'] = tb.WILLR(high_price, low_price, close_price, timeperiod=14)
    asset['AD'] = tb.AD(high_price, low_price, close_price, volume)
    asset['ADOSC'] = tb.ADOSC(high_price, low_price, close_price, volume, fastperiod=3, slowperiod=10)
    asset['OBC'] = tb.OBV(close_price, volume)
    asset['ATR'] = tb.ATR(high_price, low_price, close_price, timeperiod=14)
    asset['AVGPRICE'] = tb.AVGPRICE(open_price, high_price, low_price, close_price)
    asset['MEDPRICE'] = tb.MEDPRICE(high_price, low_price)
    asset['TYPPRICE'] = tb.TYPPRICE(high_price, low_price, close_price)
    asset['WCLPRICE'] = tb.WCLPRICE(high_price, low_price, close_price)

    float_columns = asset.select_dtypes(include=['float64']).columns
    asset[float_columns] = asset[float_columns].astype('float32')

    primary_features = ['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']
    asset[primary_features] = scaler.fit_transform(asset[primary_features])
    
    secondary_feature = ['Vol', 'upper_shadow', 'lower_shadow', 'upper_ratio', 'lower_ratio', 
                         'Average_volume', 'Regret', 'MA5', 'MA15', 'MA30', 'MA60', 'VMA5', 
                         'VMA15', 'VMA30', 'VMA60', 'RV5', 'RV15', 'RV30', 'RV60', 'ADX', 'ADXR',
                         'APO', 'AROONOSC', 'BOP', 'CCI', 'DX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM',
                         'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'RSI', 'TRIX', 'ULTOSC', 'WILLR',
                         'AD', 'ADOSC', 'OBC', 'ATR', 'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE']
    for feature_name in secondary_feature:
        asset.loc[np.isinf(asset[feature_name]), feature_name] = asset[~np.isinf(asset[feature_name])][feature_name].mean()
        asset.loc[np.isnan(asset[feature_name]), feature_name] = asset[~np.isnan(asset[feature_name])][feature_name].mean()
    asset.fillna(0, inplace=True)
    asset[secondary_feature] = scaler.fit_transform(asset[secondary_feature])
    
    return asset

def process_time_series_feature(data, verbose=False):
    processed_assets = []
    for aid in range(14):
        if verbose:
            print(f"Processing: {asset_names[aid]}")
        asset = data[data['Asset_ID'] == aid].copy(deep=True)
        asset = process_one_asset(asset)
        processed_assets.append(asset)
    
    if verbose:
        print("Concat processed single asset data")
    processed_assets = pd.concat(processed_assets)
    processed_assets.sort_values(by='index', inplace=True)
    processed_assets.drop(['index'], axis=1, inplace=True)

    if verbose:
        print("One-Hot encoding")

    onehot = pd.get_dummies(processed_assets['Asset_ID'], dtype=int)
    for i in range(14):
        if i not in onehot.columns:
            onehot[i] = 0
    onehot = onehot.reindex(columns=list(range(14)))
    onehot.columns = [asset_names[aid] for aid in onehot.columns]
    # processed_assets = pd.concat([processed_assets, onehot], axis=1)
    processed_assets = processed_assets.join(onehot)
    float_columns = processed_assets.select_dtypes(include=['float64']).columns
    processed_assets[float_columns] = processed_assets[float_columns].astype('float32')
    
    return processed_assets

def train_loop(dataloader, net, loss_fn, optimizer, device):
    running_loss = 0
    current = 0
    net.train()

    with tqdm(dataloader) as t:
        for batch, (X, y, num_assets) in enumerate(t):
            batch_size = X.shape[0]
            X = X.to(device)
            y = y.to(device)
            num_assets = num_assets.to(device)
            mask = torch.arange(14).expand(batch_size, 14).to(device)
            mask = mask < num_assets.unsqueeze(1)

            y_pred = net(X, mask)
            y_pred = y_pred.view(batch_size, 14)
            loss = loss_fn(y_pred, y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = (batch_size * loss.item() + running_loss * current) / (batch_size + current)
            current += batch_size
            t.set_postfix({'train loss':running_loss})
    
    return running_loss

def val_loop(dataloader, net, loss_fn, device):
    running_loss = 0.0
    current = 0
    net.eval()
    
    with torch.no_grad():
        with tqdm(dataloader) as t:
            for batch, (X, y, num_assets) in enumerate(t):
                batch_size = X.shape[0]
                X = X.to(device)
                y = y.to(device)
                num_assets = num_assets.to(device)
                mask = torch.arange(14).expand(batch_size, 14).to(device)
                mask = mask < num_assets.unsqueeze(1)

                y_pred = net(X, mask)
                y_pred = y_pred.view(batch_size, 14)
                loss = loss_fn(y_pred, y, mask)
                running_loss = (batch_size * loss.item() + running_loss * current) / (batch_size + current)
                t.set_postfix({'val loss':running_loss})
                
    return running_loss

def weighted_mean(x, w):
    return np.sum(x*w)/np.sum(w)

def weighted_cov(x, y, w):
    mean_x = weighted_mean(x, w)
    mean_y = weighted_mean(y, w)
    return weighted_mean((x-mean_x)*(y-mean_y), w)

def weighted_corr(x, y, w):
    return weighted_cov(x, y, w)/np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))