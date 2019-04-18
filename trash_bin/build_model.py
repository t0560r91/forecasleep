import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
from functions import (datespace, timespace, aggregate_hr, input_time_scaler,
                       parse_time, parse_date, parse_datetime,
                       format_datetime_i, format_datetime_h,
                       reformat_datetime_hi, reformat_datetime_ih,
                       get_deep, get_light, get_rem, get_wake,
                       import_concat, expand_input_time, get_input_bed,
                       get_delta, get_p_day, get_diff, get_avg, get_var,
                    )

from collections import defaultdict
from classes import TimeScaler, AvgRatioFiller, MatrixPipeline, OneHotEncoder, ZeroFiller

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline


# import
raw_all_sleep = pd.read_csv('data/sleep_archive.csv')
raw_all_sleep.columns = ['start', 'end', 'asleep','awake','awakening','bed',' rem','light','deep']
raw_all_sleep['start'] = raw_all_sleep['start'].apply(parse_datetime)
raw_all_sleep['end'] = raw_all_sleep['end'].apply(parse_datetime)
raw_all_sleep.drop_duplicates(inplace=True)
raw_all_sleep.sort_values('start', inplace=True)
raw_all_sleep.reset_index(inplace=True, drop=True)


# separate sleep & nap
nap_mask = \
(raw_all_sleep['start'].apply(lambda x: x.time()) > datetime(1,1,1,11,0).time()) &\
(raw_all_sleep['start'].apply(lambda x: x.time()) < datetime(1,1,1,16,0).time()) |\
(raw_all_sleep['asleep'] < 180)

# raw_nap
raw_nap = raw_all_sleep.loc[nap_mask, :].copy()
raw_nap.reset_index(inplace=True, drop=True)



sel_nap = pd.DataFrame()
sel_nap['date'] = raw_nap['start'].apply(lambda x: x.date())
sel_nap['nap'] = raw_nap['asleep']


# raw_sleep
raw_sleep = raw_all_sleep.loc[~nap_mask, :].copy()
raw_sleep.reset_index(inplace=True, drop=True)

# sel_sleep
sel_sleep = raw_sleep[['start','end','bed','deep']]


# import and define user input start and end time
with open('/Users/Sehokim/capstone/data/start.pkl', 'rb') as s:
    raw_input_start = pickle.load(s)
with open('/Users/Sehokim/capstone/data/end.pkl', 'rb') as s:
    raw_input_end = pickle.load(s)

# apn_sleep
input_start = expand_input_time(raw_input_start)
input_end =  expand_input_time(raw_input_end)
input_bed = get_input_bed(input_start, input_end)
input_list = [input_start, input_end, input_bed]
input_dict = defaultdict()
for k, v in zip(sel_sleep.columns, input_list):
    input_dict[k] = v
    
input_df = pd.DataFrame(input_dict, index=[len(sel_sleep)])
apn_sleep = pd.concat([sel_sleep, input_df], axis=0, sort=False)

# exp_sleep
exp_sleep = pd.DataFrame()
exp_sleep['date'] = apn_sleep['end'].apply(lambda x: x.date()) - timedelta(days=1)
exp_sleep['day'] = exp_sleep['date'].apply(lambda x: x.weekday())
exp_sleep['start'] = apn_sleep['start']
exp_sleep['end'] = apn_sleep['end']
exp_sleep['bed'] = apn_sleep['bed']
exp_sleep['deep'] = apn_sleep['deep']


exp_sleep['delta'] = get_delta(apn_sleep)
for i in range(7):
    i += 1
    exp_sleep[f'p{i}'] = get_p_day(exp_sleep, i)
exp_sleep['p1_diff'] = get_diff(exp_sleep, 'p1')
exp_sleep['p3_avg'] = get_avg(exp_sleep, 3)
exp_sleep['p7_avg'] = get_avg(exp_sleep, 7)
exp_sleep['p3_var'] = get_var(exp_sleep, 3)
exp_sleep['p7_var'] = get_var(exp_sleep, 7)
exp_sleep['p3_diff'] = get_diff(exp_sleep, 'p3_avg')
exp_sleep['p7_diff'] = get_diff(exp_sleep, 'p7_avg')

merged_sleep = exp_sleep.merge(sel_nap, on='date', how='left')

branches = [
    ('start', TimeScaler()), 
    ('end', TimeScaler()),
    ('day', OneHotEncoder()),
    ('p1', TimeScaler()),
    ('p2', TimeScaler()),
    ('p3', TimeScaler()),
    ('p4', TimeScaler()),
    ('p5', TimeScaler()),
    ('p6', TimeScaler()),
    ('p7', TimeScaler()),
    ('p3_avg', TimeScaler()),
    ('p7_avg', TimeScaler()),
    ('nap', ZeroFiller()),
    ('deep', AvgRatioFiller(merged_sleep['bed']))
]

mp = MatrixPipeline(branches)
mp.fit(merged_sleep)
Xy = mp.transform(merged_sleep)
Xy = Xy[7:].copy()
Xy.reset_index(inplace=True)
Xy.drop('index', axis=1, inplace=True)

y = Xy.pop('deep')
Xy.pop('date')
X = Xy

# get regularization strength
def ridge_cvmse(a,X,y):
    kf = KFold(5, shuffle=True)
    ridge_mse = []
    for train, test in kf.split(X):
        ridge = Ridge(a)
        ridge.fit(X.loc[train], y.loc[train])
        y_ = ridge.predict(X.loc[test])
        ridge_mse.append(mean_squared_error(y.loc[test], y_))

    return np.mean(ridge_mse)

a_space = np.logspace(np.log10(0.000001), np.log10(1000000), num=500)

mses = []
for a in a_space:
    mses.append(round(ridge_cvmse(a, X,y), 2))
    
a = a_space[np.argmin(mses)]


# build model and get prediction
Xtest = X.iloc[:-1, :].values
ytest = y.iloc[:-1].values
Xtoday = X.iloc[-1, :].values.reshape(1,-1)
ridge = Ridge(a)
ridge.fit(Xtest, ytest)

y_ = ridge.predict(Xtoday)[0]
output = f'Estimated Deep Sleep Minutes: {y_:.2f}'

with open('/Users/Sehokim/capstone/data/prediction.pkl', 'wb') as pred:
    pickle.dump(output, pred)
