import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functions import (datespace, timespace, aggregate_hr, input_time_scaler,
                       parse_time, parse_date, parse_datetime,
                       format_datetime_i, format_datetime_h,
                       reformat_datetime_hi, reformat_datetime_ih,
                       get_deep, get_light, get_rem, get_wake,
                       import_concat, expand_input_time, get_input_bed,
                       get_delta, get_p_day, get_diff, get_avg, get_var)

from collections import defaultdict
from classes import TimeScaler, AvgRatioFiller, MatrixPipeline, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline



# user = 'Beth'
# user = 'SE'
# user = 'SK'
user = 'BW'
# user = 'SeheeKim'
path = f'/Users/Sehokim/{user}/user-site-export/'
start_date = '2018-05-04'
end_date = datetime.strftime(datetime.now(), '%Y-%m-%d')
dates = datespace(start_date, end_date)

print(f'\
      User: {user}\n\
Start Date: {start_date}\n\
  End Date: {end_date}\n\
  Duration: {len(dates)} days')


# import sleep data
raw_sleep = import_concat(path, 'sleep', dates)


# sleep_df
archive_sleep_df = pd.DataFrame()
archive_sleep_df['Start Time'] = raw_sleep['startTime'].apply(reformat_datetime_hi)
archive_sleep_df['End Time'] = raw_sleep['endTime'].apply(reformat_datetime_hi)
archive_sleep_df['Minutes Asleep'] = raw_sleep['minutesAsleep']
archive_sleep_df['Minutes Awake'] = raw_sleep['minutesAwake']
archive_sleep_df['Number of Awakenings'] = raw_sleep['levels'].apply(get_wake)
archive_sleep_df['Time in Bed'] = raw_sleep['timeInBed']
archive_sleep_df['Minutes REM Sleep'] = raw_sleep['levels'].apply(get_rem)
archive_sleep_df['Minutes Light Sleep'] = raw_sleep['levels'].apply(get_light)
archive_sleep_df['Minutes Deep Sleep'] = raw_sleep['levels'].apply(get_deep)
archive_sleep_df.to_csv('./data/sleep_archive2.csv', columns=archive_sleep_df.columns, index=None)