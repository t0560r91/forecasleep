import numpy as np
import pandas as pd
import fitbit
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from functions import (
    datespace, parse_datetime, expand_input_time, get_input_bed, get_delta, get_delta_scale, get_p_day, get_diff, get_avg, get_var, stitch_drop_append2, estimator_cv_scores)
from classes import (
    TimeScaler, AvgRatioFiller, MatrixPipeline, OneHotEncoder, ZeroFiller, ChainTransformer, StandardScaler, AvgFiller)





# import tokens
with open('/Users/Sehokim/capstone/data/access_token.pkl', 'rb') as a:
    access = pickle.load(a)
with open('/Users/Sehokim/capstone/data/refresh_token.pkl', 'rb') as r:
    refresh = pickle.load(r)
client_id = '22DGXK'
client_secret = '6bb599aceb4be6c3b3caa2163247ffee'

# authenticate with token
auth_client = fitbit.Fitbit(
    client_id, 
    client_secret, 
    access_token=access, 
    refresh_token=refresh)

#define date range
start_date = '2019-02-23'
end_date = datetime.strftime(datetime.now(), '%Y-%m-%d')
dates = datespace(start_date, end_date)



# import sleep data
res = auth_client.time_series('sleep', base_date=start_date, end_date=end_date)
df = pd.DataFrame(res['sleep'])

# select columns from raw sleep data
raw_all_sleep = pd.DataFrame()
raw_all_sleep['start'] = df['startTime'].apply(parse_datetime)
raw_all_sleep['end'] = df['endTime'].apply(parse_datetime)
raw_all_sleep['bed'] = df['timeInBed']
raw_all_sleep['asleep'] = df['minutesAsleep']
raw_all_sleep['awake'] = df['minutesAwake']
raw_all_sleep['effic'] = df['efficiency']
# sleep['awakening'] = df['levels'].apply(get_wake)
# sleep['Minutes REM Sleep'] = raw_sleep['levels'].apply(get_rem)
# sleep['Minutes Light Sleep'] = raw_sleep['levels'].apply(get_light)
# sleep['Minutes Deep Sleep'] = raw_sleep['levels'].apply(get_deep)
raw_all_sleep.sort_values('start', inplace=True)
raw_all_sleep.reset_index(inplace=True, drop=True)
sti_all_sleep = stitch_drop_append2(raw_all_sleep)


# mask nap and sleep
sync_sleep_mask = \
((sti_all_sleep['start'].apply(lambda x: x.time()) >= datetime(1,1,1,17,0).time()) |\
(sti_all_sleep['start'].apply(lambda x: x.time()) < datetime(1,1,1,5,0).time())) &\
(sti_all_sleep['asleep'] >= 180)

sync_nap_mask = \
((sti_all_sleep['start'].apply(lambda x: x.time()) >= datetime(1,1,1,22,0).time()) |\
(sti_all_sleep['start'].apply(lambda x: x.time()) < datetime(1,1,1,10,0).time())) &\
(sti_all_sleep['asleep'] < 180)

async_sleep_mask = \
(sti_all_sleep['start'].apply(lambda x: x.time()) > datetime(1,1,1,5,0).time()) &\
(sti_all_sleep['start'].apply(lambda x: x.time()) < datetime(1,1,1,17,0).time()) &\
(sti_all_sleep['asleep'] >= 180)

async_nap_mask = \
(sti_all_sleep['start'].apply(lambda x: x.time()) > datetime(1,1,1,10,0).time()) &\
(sti_all_sleep['start'].apply(lambda x: x.time()) < datetime(1,1,1,22,0).time()) &\
(sti_all_sleep['asleep'] < 180)



# raw_nap
raw_nap = sti_all_sleep.loc[async_sleep_mask, :].copy()
raw_nap.reset_index(inplace=True, drop=True)
# sel_nap
sel_nap = pd.DataFrame()
sel_nap['date'] = raw_nap['start'].apply(lambda x: x.date())
sel_nap['nap'] = raw_nap['asleep']



# raw_sleep
raw_sleep = sti_all_sleep.loc[sync_sleep_mask, :].copy()
raw_sleep.reset_index(inplace=True, drop=True)
# sel_sleep
sel_sleep = raw_sleep[['start','end','bed','effic']]

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
def get_sum(x):
    mask1 = exp_sleep.date < x
    mask2 = exp_sleep.date > x-timedelta(days=3)
    return exp_sleep.loc[mask1 & mask2, 'bed'].mean()

exp_sleep = pd.DataFrame()
exp_sleep['date'] = apn_sleep['end'].apply(lambda x: x.date()) - timedelta(days=1)
exp_sleep['day'] = exp_sleep['date'].apply(lambda x: x.weekday())
exp_sleep['start'] = apn_sleep['start']
exp_sleep['end'] = apn_sleep['end']
exp_sleep['bed'] = apn_sleep['bed']
exp_sleep['effic'] = apn_sleep['effic']
exp_sleep['delta'] = get_delta_scale(apn_sleep)

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
exp_sleep['p3_sum'] = exp_sleep['date'].apply(get_sum)



# merge sleep and nap back together
merged_sleep = exp_sleep.merge(sel_nap, on='date', how='left')
# merged = merged_sleep.merge(heart, on='date', how='left')\
# .merge(activity, on='date', how='left')\



# transfrom each columns for Xy
time_sc = ChainTransformer([TimeScaler()])
zero_sc = ChainTransformer([ZeroFiller()])
avg_sc = ChainTransformer([AvgFiller()])

branches = [
    ('start', time_sc), 
    ('end', time_sc),
    ('day', OneHotEncoder()),
    ('p1', time_sc),
    ('p2', time_sc),
    ('p3', time_sc),
    ('p4', time_sc),
    ('p5', time_sc),
    ('p6', time_sc),
    ('p7', time_sc),
    ('p3_avg', time_sc),
    ('p7_avg', time_sc),
    ('nap', zero_sc),
    ('get_top5', avg_sc),
    ('get_bottom5', avg_sc),
    ('mean', avg_sc),
    ('p3_sum', avg_sc)]

mp = MatrixPipeline(branches)
mp.fit(merged_sleep)
Xy = mp.transform(merged_sleep)
Xy = Xy[7:].copy()
Xy.reset_index(inplace=True, drop=True)
# Xy.drop('index', axis=1, inplace=True)
Xy.pop('date')
y = Xy.pop('effic')
X = Xy



# build random forest
rfr = RandomForestRegressor(1000)
rfr.fit(X[:len(X)-1],y[:len(X)-1])
y_ = rfr.predict(X[len(X)-1:])[0]
avg_p7_effic = np.mean(y[-8:-1])

result = (
f'Estimated sleep efficiency for the night of {datetime.today().date()}: {y_:10.2f} out of 100',
f'Average sleep efficiency for the past 7 days: {avg_p7_effic:20.2f} out of 100')

# export the result
with open('/Users/Sehokim/capstone/data/prediction.pkl', 'wb') as p:
    pickle.dump(result, p)