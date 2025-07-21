import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from src.utils import smape, vsmape

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format

# Set data directory
DATA_DIR = 'data/'

# Load data
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), encoding='utf-8')
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), encoding='utf-8')
df_census = pd.read_csv(os.path.join(DATA_DIR, 'census_starter.csv'), encoding='utf-8')
reaveal_test = pd.read_csv(os.path.join(DATA_DIR, 'revealed_test.csv'), encoding='utf-8')
df_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), encoding='utf-8')
coords = pd.read_csv(os.path.join(DATA_DIR, 'cfips_location.csv'), encoding='utf-8')

# Remove November and December 2022 from test set
invalid_idx = (test.first_day_of_month == '2022-11-01') | (test.first_day_of_month == '2022-12-01')
test = test.loc[~invalid_idx, :]

# Merge train and revealed_test, sort
train = pd.concat([train, reaveal_test]).sort_values(by=['cfips', 'first_day_of_month']).reset_index(drop=True)
train['istest'] = 0
test['istest'] = 1
raw = pd.concat((train, test)).sort_values(['cfips', 'row_id']).reset_index(drop=True)
raw = raw.merge(coords.drop("name", axis=1), on="cfips")

# Type conversion and basic features
def preprocess_raw(raw):
    raw['state_i1'] = raw['state'].astype('category')
    raw['county_i1'] = raw['county'].astype('category')
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['county'] = raw.groupby('cfips')['county'].ffill()
    raw['state'] = raw.groupby('cfips')['state'].ffill()
    raw["dcount"] = raw.groupby(['cfips'])['row_id'].cumcount()
    raw['county_i'] = (raw['county'] + raw['state']).factorize()[0]
    raw['state_i'] = raw['state'].factorize()[0]
    raw['scale'] = (raw['first_day_of_month'] - raw['first_day_of_month'].min()).dt.days
    raw['scale'] = raw['scale'].factorize()[0]
    raw['year'] = raw['first_day_of_month'].dt.year
    return raw
raw = preprocess_raw(raw)

# Outlier correction
outliers = []
for o in tqdm(raw.cfips.unique(), desc='Outlier Correction'):
    indices = (raw['cfips'] == o)
    tmp = raw.loc[indices].copy().reset_index(drop=True)
    var = tmp.microbusiness_density.values.copy()
    for i in range(40, 0, -1):
        thr = 0.20 * np.mean(var[:i])
        difa = abs(var[i] - var[i - 1])
        if (difa >= thr):
            var[:i] *= (var[i] / var[i - 1])
            outliers.append(o)
    var[0] = var[1] * 0.99
    raw.loc[indices, 'microbusiness_density'] = var

# Lag features
lag = 1
raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()
raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1
raw.loc[(raw[f'mbd_lag_{lag}'] == 0), 'dif'] = 0
raw.loc[(raw['microbusiness_density'] > 0) & (raw[f'mbd_lag_{lag}'] == 0), 'dif'] = 1
raw['dif'] = raw['dif'].abs()

# Target transformation for SMAPE
raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
raw['target'] = raw['target'] / raw['microbusiness_density'] - 1
raw.loc[raw['cfips'] == 28055, 'target'] = 0.0
raw.loc[raw['cfips'] == 48269, 'target'] = 0.0

# Last active and last target features
raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
dt = raw.loc[raw.dcount == 40].groupby('cfips')['microbusiness_density'].agg('last')
raw['lasttarget'] = raw['cfips'].map(dt)

# Feature engineering
def build_features(raw, target='microbusiness_density', target_act='active', lags=6):
    feats = []
    for lag in range(1, lags):
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')[target].shift(lag)
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[target_act].diff(lag)
        feats.append(f'mbd_lag_{lag}')
        feats.append(f'act_lag_{lag}')
    lag = 1
    for window in [2, 4, 6, 8, 10]:
        raw[f'mbd_rollmea{window}_{lag}'] = raw.groupby('cfips')[f'mbd_lag_{lag}'].transform(lambda s: s.rolling(window, min_periods=1).sum())
        feats.append(f'mbd_rollmea{window}_{lag}')
    # Merge census features
    census_columns = list(df_census.columns)
    census_columns.remove('cfips')
    raw = raw.merge(df_census, on='cfips', how='left')
    feats += census_columns
    # Merge population estimate features
    co_est = pd.read_csv(os.path.join(DATA_DIR, 'co-est2021-alldata.csv'), encoding='latin-1')
    co_est['cfips'] = co_est.STATE * 1000 + co_est.COUNTY
    co_columns = [
        'SUMLEV', 'DIVISION', 'ESTIMATESBASE2020', 'POPESTIMATE2020', 'POPESTIMATE2021',
        'NPOPCHG2020', 'NPOPCHG2021', 'BIRTHS2020', 'BIRTHS2021', 'DEATHS2020', 'DEATHS2021',
        'NATURALCHG2020', 'NATURALCHG2021', 'INTERNATIONALMIG2020', 'INTERNATIONALMIG2021',
        'DOMESTICMIG2020', 'DOMESTICMIG2021', 'NETMIG2020', 'NETMIG2021', 'RESIDUAL2020',
        'RESIDUAL2021', 'GQESTIMATESBASE2020', 'GQESTIMATES2020', 'GQESTIMATES2021',
        'RBIRTH2021', 'RDEATH2021', 'RNATURALCHG2021', 'RINTERNATIONALMIG2021',
        'RDOMESTICMIG2021', 'RNETMIG2021'
    ]
    raw = raw.merge(co_est, on='cfips', how='left')
    feats += co_columns
    return raw, feats

raw, feats = build_features(raw, 'target', 'active', lags=9)
features = ['state_i'] + feats + ['lng', 'lat', 'scale']

# Rotated latitude/longitude features
def rot(df):
    for angle in [15, 30, 45]:
        df[f'rot_{angle}_x'] = (np.cos(np.radians(angle)) * df['lat']) + (np.sin(np.radians(angle)) * df['lng'])
        df[f'rot_{angle}_y'] = (np.cos(np.radians(angle)) * df['lat']) - (np.sin(np.radians(angle)) * df['lng'])
    return df
raw = rot(raw)
features += ['rot_15_x', 'rot_15_y', 'rot_30_x', 'rot_30_y', 'rot_45_x', 'rot_45_y']

# Set some features to NaN for certain years to prevent leakage
for idx in raw.year.unique():
    if idx == 2019:
        for col in [f'pct_it_workers_{idx}', f'pct_it_workers_{idx+1}', f'pct_it_workers_{idx+2}',
                    f'median_hh_inc_{idx}', f'median_hh_inc_{idx+1}', f'median_hh_inc_{idx+2}',
                    f'pct_bb_{idx}', f'pct_bb_{idx+1}', f'pct_bb_{idx+2}',
                    f'pct_college_{idx}', f'pct_college_{idx+1}', f'pct_college_{idx+2}',
                    f'pct_foreign_born_{idx}', f'pct_foreign_born_{idx+1}', f'pct_foreign_born_{idx+2}',
                    'ESTIMATESBASE2020', 'POPESTIMATE2020', 'POPESTIMATE2021', 'NPOPCHG2020', 'NPOPCHG2021',
                    'BIRTHS2020', 'BIRTHS2021', 'DEATHS2020', 'DEATHS2021', 'NATURALCHG2020', 'NATURALCHG2021',
                    'INTERNATIONALMIG2020', 'INTERNATIONALMIG2021', 'DOMESTICMIG2020', 'DOMESTICMIG2021',
                    'NETMIG2020', 'NETMIG2021', 'RESIDUAL2020', 'RESIDUAL2021', 'GQESTIMATESBASE2020',
                    'GQESTIMATES2020', 'GQESTIMATES2021', 'RBIRTH2021', 'RDEATH2021', 'RNATURALCHG2021',
                    'RINTERNATIONALMIG2021', 'RDOMESTICMIG2021', 'RNETMIG2021']:
            if col in raw.columns:
                raw.loc[raw.year == idx, col] = np.nan
    if idx == 2020:
        for col in [f'pct_it_workers_{idx}', f'pct_it_workers_{idx+1}',
                    f'median_hh_inc_{idx}', f'median_hh_inc_{idx+1}',
                    f'pct_bb_{idx}', f'pct_bb_{idx+1}',
                    f'pct_college_{idx}', f'pct_college_{idx+1}',
                    f'pct_foreign_born_{idx}', f'pct_foreign_born_{idx+1}',
                    'ESTIMATESBASE2020', 'POPESTIMATE2020', 'POPESTIMATE2021', 'NPOPCHG2020', 'NPOPCHG2021',
                    'BIRTHS2020', 'BIRTHS2021', 'DEATHS2020', 'DEATHS2021', 'NATURALCHG2020', 'NATURALCHG2021',
                    'INTERNATIONALMIG2020', 'INTERNATIONALMIG2021', 'DOMESTICMIG2020', 'DOMESTICMIG2021',
                    'NETMIG2020', 'NETMIG2021', 'RESIDUAL2020', 'RESIDUAL2021', 'GQESTIMATESBASE2020',
                    'GQESTIMATES2020', 'GQESTIMATES2021', 'RBIRTH2021', 'RDEATH2021', 'RNATURALCHG2021',
                    'RINTERNATIONALMIG2021', 'RDOMESTICMIG2021', 'RNETMIG2021']:
            if col in raw.columns:
                raw.loc[raw.year == idx, col] = np.nan
    if idx == 2021:
        for col in [f'pct_it_workers_{idx}', f'median_hh_inc_{idx}', f'pct_bb_{idx}', f'pct_college_{idx}',
                    f'pct_foreign_born_{idx}', 'POPESTIMATE2021', 'NPOPCHG2021', 'BIRTHS2021', 'DEATHS2021',
                    'NATURALCHG2021', 'INTERNATIONALMIG2021', 'DOMESTICMIG2021', 'NETMIG2021', 'RESIDUAL2021',
                    'GQESTIMATES2021', 'RBIRTH2021', 'RDEATH2021', 'RNATURALCHG2021', 'RINTERNATIONALMIG2021',
                    'RDOMESTICMIG2021', 'RNETMIG2021']:
            if col in raw.columns:
                raw.loc[raw.year == idx, col] = np.nan

# Model stacking and training
from sklearn.ensemble import VotingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer

# Model definitions (as in notebook)
def get_model():
    params = {
        'n_iter': 200,
        'verbosity': -1,
        'objective': 'l1',
        'random_state': 42,
        'colsample_bytree': 0.8841279649367693,
        'colsample_bynode': 0.10142964450634374,
        'max_depth': 8,
        'learning_rate': 0.013647749926797374,
        'lambda_l1': 1.8386216853616875,
        'lambda_l2': 7.557660410418351,
        'num_leaves': 61,
        "seed": 42,
        'min_data_in_leaf': 213
    }
    lgb_model = lgb.LGBMRegressor(**params)
    xgb_model = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        tree_method="hist",
        n_estimators=795,
        learning_rate=0.0075,
        max_leaves=17,
        subsample=0.50,
        colsample_bytree=0.50,
        max_bin=4096,
        n_jobs=2,
    )
    cat_model = cat.CatBoostRegressor(
        iterations=2000,
        loss_function="MAPE",
        verbose=0,
        grow_policy='SymmetricTree',
        learning_rate=0.035,
        l2_leaf_reg=0.2,
        colsample_bylevel=0.8,
        subsample=0.70,
        max_bin=4096,
        max_depth=5,
    )
    knn_model = Pipeline([
        ('imputer', KNNImputer(n_neighbors=2)),
        ('knn', KNeighborsRegressor(5))
    ])
    return VotingRegressor([
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model),
        ('knn', knn_model)
    ])

def base_models():
    params = {
        'n_iter': 300,
        'boosting_type': 'dart',
        'verbosity': -1,
        'objective': 'l1',
        'random_state': 42,
        'colsample_bytree': 0.8841279649367693,
        'colsample_bynode': 0.10142964450634374,
        'max_depth': 8,
        'learning_rate': 0.003647749926797374,
        'lambda_l2': 0.5,
        'num_leaves': 61,
        "seed": 42,
        'min_data_in_leaf': 213,
        'device': 'gpu'
    }
    lgb_model = lgb.LGBMRegressor(**params)
    xgb_model = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        tree_method="gpu_hist",
        n_estimators=795,
        learning_rate=0.0075,
        max_leaves=17,
        subsample=0.50,
        colsample_bytree=0.50,
        max_bin=4096,
        n_jobs=2)
    cat_model = cat.CatBoostRegressor(
        iterations=2000,
        loss_function="MAPE",
        verbose=0,
        grow_policy='SymmetricTree',
        learning_rate=0.035,
        colsample_bylevel=0.8,
        max_depth=5,
        l2_leaf_reg=0.2,
        subsample=0.70,
        max_bin=4096,
    )
    models = {}
    models['xgb'] = xgb_model
    models['lgbm'] = lgb_model
    models['cat'] = cat_model
    return models

# Stacking training as in notebook
ACT_THR = 150
MONTH_1 = 39
MONTH_last = 41
raw['ypred_last'] = np.nan
raw['ypred'] = np.nan
raw['k'] = 1.
raw['microbusiness_density'].fillna(2, inplace=True)

for TS in range(MONTH_1, MONTH_last):
    print(f'TS: {TS}')
    models = base_models()
    model0 = models['xgb']
    model1 = models['lgbm']
    model2 = models['cat']
    train_indices = (raw.istest == 0) & (raw.dcount < TS) & (raw.dcount >= 1) & (raw.lastactive > ACT_THR)
    valid_indices = (raw.istest == 0) & (raw.dcount == TS)
    model0.fit(raw.loc[train_indices, features], raw.loc[train_indices, 'target'].clip(-0.002, 0.006))
    model1.fit(raw.loc[train_indices, features], raw.loc[train_indices, 'target'].clip(-0.002, 0.006))
    model2.fit(raw.loc[train_indices, features], raw.loc[train_indices, 'target'].clip(-0.002, 0.006))
    tr_pred0 = model0.predict(raw.loc[train_indices, features])
    tr_pred1 = model1.predict(raw.loc[train_indices, features])
    tr_pred2 = model2.predict(raw.loc[train_indices, features])
    train_preds = np.column_stack((tr_pred0, tr_pred1, tr_pred2))
    meta_model = get_model()
    meta_model.fit(train_preds, raw.loc[train_indices, 'target'].clip(-0.002, 0.006))
    val_preds0 = model0.predict(raw.loc[valid_indices, features])
    val_preds1 = model1.predict(raw.loc[valid_indices, features])
    val_preds2 = model2.predict(raw.loc[valid_indices, features])
    valid_preds = np.column_stack((val_preds0, val_preds1, val_preds2))
    ypred = meta_model.predict(valid_preds)
    raw.loc[valid_indices, 'k'] = ypred + 1
    raw.loc[valid_indices, 'k'] = raw.loc[valid_indices, 'k'] * raw.loc[valid_indices, 'microbusiness_density']
    lastval = raw.loc[raw.dcount == TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()['microbusiness_density']
    dt = raw.loc[raw.dcount == TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']
    df = raw.loc[raw.dcount == (TS + 1), ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(drop=True)
    df['pred'] = df['cfips'].map(dt)
    df['lastval'] = df['cfips'].map(lastval)
    df.loc[df['lastactive'] <= ACT_THR, 'pred'] = df.loc[df['lastactive'] <= ACT_THR, 'lastval']
    raw.loc[raw.dcount == (TS + 1), 'ypred'] = df['pred'].values
    raw.loc[raw.dcount == (TS + 1), 'ypred_last'] = df['lastval'].values
    print('Last Value SMAPE:', smape(df['microbusiness_density'], df['lastval']))
    print('SMAPE:', smape(df['microbusiness_density'], df['pred']))
    print()

# Continue prediction for future months (TS >= 41) using the trained models
for TS in range(41, 47):
    print(f'TS: {TS}')
    valid_indices = (raw.dcount == TS)
    # Use the last trained base and meta models from stacking
    val_preds0 = model0.predict(raw.loc[valid_indices, features])
    val_preds1 = model1.predict(raw.loc[valid_indices, features])
    val_preds2 = model2.predict(raw.loc[valid_indices, features])
    valid_preds = np.column_stack((val_preds0, val_preds1, val_preds2))
    ypred = meta_model.predict(valid_preds)
    raw.loc[valid_indices, 'k'] = ypred + 1
    raw.loc[valid_indices, 'k'] = raw.loc[valid_indices, 'k'] * raw.loc[valid_indices, 'microbusiness_density']
    lastval = raw.loc[raw.dcount == TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()['microbusiness_density']
    dt = raw.loc[raw.dcount == TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']
    df = raw.loc[raw.dcount == (TS + 1), ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(drop=True)
    df['pred'] = df['cfips'].map(dt)
    df['lastval'] = df['cfips'].map(lastval)
    df.loc[df['lastactive'] <= ACT_THR, 'pred'] = df.loc[df['lastactive'] <= ACT_THR, 'lastval']
    raw.loc[raw.dcount == (TS + 1), 'ypred'] = df['pred'].values
    raw.loc[raw.dcount == (TS + 1), 'microbusiness_density'] = df['pred'].values
    raw.loc[raw.dcount == (TS + 1), 'ypred_last'] = df['lastval'].values
    print(f'TS: {TS}')
    print('Last Value SMAPE:', smape(raw.loc[raw.dcount == TS, 'microbusiness_density'], raw.loc[raw.dcount == TS, 'ypred_last']))
    print('SMAPE:', smape(raw.loc[raw.dcount == TS, 'microbusiness_density'], raw.loc[raw.dcount == TS, 'ypred']))
    print()

# Generate the final submission file as in the notebook
# Select the test set rows and output the required columns
submission = raw.loc[raw.istest == 1, ['row_id', 'microbusiness_density']].copy()
submission.to_csv('submission.csv', index=False)
print('Submission file saved as submission.csv')
