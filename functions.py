from datetime import datetime, timedelta    
import numpy as np
import pandas as pd
from classes import TimeScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


### parser(col)
### getter(df)
### transformer(srs or df)


# GENERAL
#############################################
def import_concat(path, table, dates):
    """
    import multiple files and concatenate them into a pandas df
    """
    output = pd.DataFrame()
    for date in dates:
        try:
            each = pd.read_json(f'{path}{table}-{date}.json')
            output = pd.concat([output, each])
        except:
            continue
    return output


# DATE & TIME
#####################################################
def datespace(start, end, step=1):
    """
    INPUT:
        start: date string in '%Y-%m-%d'
        end: date string in '%Y-%m-%d'

    OUTPUT:
        list of datetime.date objects ranging from the given start and end date

    """
    if start <= end:
        a = datetime.strptime(start, '%Y-%m-%d')
        z = datetime.strptime(end, '%Y-%m-%d')
        result = []
        result.append(a.date())
        while z > a:
            a += timedelta(days=step)
            if a <= z:
                result.append(a.date())
        return result
    else: 
        raise ValueError('Start Date cannot be before End Date.')


def scale_time(time):
    """
    "Scale and split time to sin_cos_time"
    
    INPUT:
        start: time string in '%H:%M'
    
    OUTPUT:
        sin_time, cos_time
    """
    td = timedelta(hours=time.hour, minutes=time.minute)
    in_minutes = td.seconds//60
    scaler = 2*np.pi/1440
    halfway_scaled = in_minutes * scaler
    return np.sin(halfway_scaled), np.cos(halfway_scaled)


def parse_datetime(time_str, out_format='datetime'):
    """
    "reads time_str in any format and writes a datetime, date, or time obejct"
    
    INPUT:
        time_str: time string in any format
        out_format: datetime, date, or time
    
    OUTPUT:
        parsed time_string
    """
    in_formats = ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%d %H:%M:%S.%f', 
                  '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', 
                  '%Y-%m-%dT%I:%M%p', '%Y-%m-%d %I:%M%p',
                  '%Y-%m-%d']
    for f in in_formats:
        try:
            if out_format == 'datetime':
                return datetime.strptime(time_str, f)
            elif out_format == 'date':
                return datetime.strptime(time_str, f).date()
            elif out_format == 'time':
                return datetime.strptime(time_str, f).time()
        except:
            continue


def reformat_time_string(time_str):
    """
    "reads time_str in any format and reformats it to 24 hour cycle or 12 hour cycle 
    which ever was not the original format."

    INPUT:
        time_str: time string in any format
    
    OUTPUT:
        time_str: time string reformated
    """
    dt = parse_datetime(time_str)
    reformat = datetime.strftime(dt, '%Y-%m-%d %I:%M%p')
    if reformat != time_str:
        return reformat
    else:
        return datetime.strftime(dt, '%Y-%m-%d %H:%M:S')


# fitbit sleep specific
#############################     
def parse_stage(x, stage='deep'):
    """
    "parse a Fitbit-generated nested dictionary of sleep stage data 
    and returns a specified stage data."
    
    INPUT:
        x: a nested dictionary
        stage: sleep stage

    OUTPUT:
        minutes or counts of each specified sleep stage
    
    """
    try:
        if stage in ['deep','rem','light']:
            return x['summary'][stage]['minutes']
        elif stage == 'wake':
            return x['summary'][stage]['count']
    except:
        return np.nan   

def stitch_drop_append(df):
    """
    takes in a Fitbit style Pandas Data Frame and return a new DataFrame 
    with sleep logs that are less than 90 minutes apart from one another stitched together into a one sleep log. 
    """
    new_df=df.copy()
    later = np.where((get_delta(new_df) < 120) & (get_delta(new_df) > 0))[0]
    earlier = later - 1
    stitch_ind = list(zip(earlier, later))
    for e, l in stitch_ind:
        new_start = new_df.loc[e,'start']
        new_end = new_df.loc[l,'end']
        new_bed = (new_df.loc[l,'end'] - new_df.loc[e,'start']).seconds/60
        if ~np.isnan(new_df.loc[l,'deep'] + new_df.loc[e, 'deep']):
            new_deep = new_df.loc[l,'deep'] + new_df.loc[e, 'deep']
        else:
            new_deep = np.nan
        new_dict = {'start':new_start, 'end':new_end, 'bed':new_bed, 'deep':new_deep}

        new_df.drop([e,l], inplace=True)        
        new_df = pd.concat([new_df, pd.DataFrame(new_dict, index=[100])], sort=False)
    new_df.sort_values('start', inplace=True)
    new_df.reset_index(inplace=True, drop=True)
    return new_df


# broadcasting operators
################################################################
def get_delta(df):
    """
    takes in two datetime columns
    returns datetime column with timedelta between the two points in time.
    """
    df = df.copy()
    bracket = np.zeros(df.shape[0])
    bracket[1:] = df.loc[1:,'start'].values - df.loc[:df.shape[0]-2, 'end'].values
    return bracket/60000000000
    
def get_delta_scale(df):
    """
    takes in two datetime columns
    returns datetime column with timedelta between the two points in time cycling back to 24hous period.
    """
    df = df.copy()
    bracket = np.zeros(df.shape[0])
    bracket[1:] = df.loc[1:,'start'].values - df.loc[:df.shape[0]-2, 'end'].values
    return bracket/60000000000 % 1440

def get_p_day(df, n):
    """
    takes in two datetime columns
    returns a datetime column with time difference values.
    """
    n -= 1
    p1 = list(df.loc[:df.shape[0]-(2+n), 'start'])
    for _ in range(n+1):
        p1.insert(0, datetime(2000,1,1,0,0))
    return p1

def get_diff(df, col):
    """
    takes in two datetime columns
    returns a datetime column with time difference values
    """
    def convert_diff(x):
        if x > 900:
            x -= 1440 
        elif x < -900:
            x += 1440
        return x

    td_minute_p = df[col].apply(lambda x: timedelta(hours=x.hour, minutes=x.minute).seconds/60)
    td_minute_start = df['start'].apply(lambda x: timedelta(hours=x.hour, minutes=x.minute).seconds/60)
    raw_diff = td_minute_start - td_minute_p
    return  raw_diff.apply(convert_diff)

def get_avg(df, n):
    """
    takes in multiple datetime columns
    returns a columns with average datetime values
    """
    # generate ingredients
    scaler = TimeScaler()
    box = pd.DataFrame()
    for i in range(n):
        i += 1
        scaler.fit(df[f'p{i}'])
        item = scaler.transform(df[f'p{i}'])        
        box = pd.concat([box, item], axis = 1, sort=False)
    
    # select columns
    sin_cols = box.columns[['sin' in x for x in box.columns]]
    cos_cols = box.columns[['cos' in x for x in box.columns]]
    
    # aggregate values by columns
    avg_sin = box.loc[:, sin_cols].mean(axis=1)
    avg_cos = box.loc[:, cos_cols].mean(axis=1)
    
    # get norm for each time
    norm = np.sqrt(avg_sin**2 + avg_cos**2)
    
    # normalize
    normed_avg_sin = avg_sin/norm
    normed_avg_cos = avg_cos/norm

    return scaler.reverse(normed_avg_sin, normed_avg_cos)

def get_var(df, n):
    """
    takes in multiple datetime columns 
    returns a columns with variance among the columns
    """
    # generate ingredients
    scaler = TimeScaler()
    box = pd.DataFrame()
    for i in range(n):
        i += 1
        scaler.fit(df[f'p{i}'])
        item = scaler.transform(df[f'p{i}'])        
        box = pd.concat([box, item], axis = 1, sort=False)
    
    # select columns
    sin_cols = box.columns[['sin' in x for x in box.columns]]
    cos_cols = box.columns[['cos' in x for x in box.columns]]
    
    # aggregate values by columns
    var_sin = box.loc[:, sin_cols].std(axis=1)
    var_cos = box.loc[:, cos_cols].std(axis=1)
    
    return var_cos + var_sin
    


# fitbit HR specific
#############################################
def get_top5(x):
    """
    takes an array
    returns the top 5% values of the array
    """
    x = x.sort_values(ascending=False)
    cut = int(len(x)*0.05 // 1)
    new_x = x[:cut].copy()
    return new_x.mean()

def get_bottom5(x):
    """
    takes an array
    returns the bottom 5% values of the array
    """
    x = x.sort_values(ascending=False)
    cut = int(len(x)*0.95 // 1)
    new_x = x[cut:].copy()
    return new_x.mean()

#####################################################


# INPUT
#############################################
def expand_input_time(input_time):
    """
    takes in a time string in %H:%M formt 
    returns a datatime object with today's date appended to it. 
    """
    h = datetime.strptime(input_time,'%H:%M').hour
    m = datetime.strptime(input_time,'%H:%M').minute
    str_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    now = datetime.strptime(str_now, '%Y-%m-%d %H:%M:%S')
    if h < 24 and h > 12:
        input_time = now.replace(hour=h, minute=m, second=0, )
    else:
        input_time = now.replace(hour=h, minute=m, second=0) + timedelta(days=1)
    return input_time

def get_input_bed(input_start, input_end):
    """
    takes in two datetime objects
    returns minutes between the two points in time in integers.
    """
    td = input_end - input_start
    return td.seconds/60


# MODELING 
##########################################

def estimator_cv_scores(X, y, estimator, a=None, max_iter=None):
    """
    takes in a dataset and a target column, and an estimator 
    returns cross validated test mses, r2 scores, and train mse, r2 scores.
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    kf = KFold(5, shuffle=True)
    test_mses = []
    test_r2s = []
    train_mses = []
    train_r2s = []
    for train, test in kf.split(X):
        model = estimator(alpha=a, max_iter=max_iter)
        model.fit(X.loc[train], y.loc[train])
        ytest_ = model.predict(X.loc[test])
        ytrain_ = model.predict(X.loc[train])
        try:
            coef = model.coef_
        except:
            coef = None
        try:
            intercept = model.intercept_
        except:
            intercept = None
        test_mses.append(mean_squared_error(y.loc[test], ytest_))
        test_r2s.append(r2_score(y.loc[test], ytest_))
        train_mses.append(mean_squared_error(y.loc[train], ytrain_))
        train_r2s.append(r2_score(y.loc[train], ytrain_))
    return (np.mean(test_mses), np.mean(test_r2s), 
    np.mean(train_mses), np.mean(train_r2s), coef, intercept)

def estimator_cv_scores2(X, y, estimator, a=None):
    from sklearn.metrics import mean_squared_error, r2_score
    X = X.values.reshape(-1,1).copy()
    kf = KFold(5, shuffle=True)
    test_mses = []
    test_r2s = []
    train_mses = []
    train_r2s = []
    for train, test in kf.split(X):
        model = estimator(a)
        model.fit(X[train], y.loc[train])
        ytest_ = model.predict(X[test])
        ytrain_ = model.predict(X[train])
        try:
            coef = model.coef_
        except:
            coef = None
        try:
            intercept = model.intercept_
        except:
            intercept = None
        test_mses.append(mean_squared_error(y.loc[test], ytest_))
        test_r2s.append(r2_score(y.loc[test], ytest_))
        train_mses.append(mean_squared_error(y.loc[train], ytrain_))
        train_r2s.append(r2_score(y.loc[train], ytrain_))
    return (np.mean(test_mses), np.mean(test_r2s), 
    np.mean(train_mses), np.mean(train_r2s), coef, intercept)
