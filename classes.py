# time to timedleta to minutes
# minutes to sin-cos scale
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from math import floor

class TimeScaler:
    """
    "Scales and splits time values into sin and cos values
    for time value continuity (clockwise) reasons."
    
    INPUT:
        series: a target datetime(or time) column in datetime(or time) object in pd-s
    
    OUTPUT:
        result: a transformed sin-column and a cos-column in pd-df
    """
    def __init__(self):
        self.sin = None
        self.cos = None
        
    def fit(self, X=None, y=None):
        self.X = X
        
    def transform(self, X):
        """
        time to sin, cos
        """
        result = pd.DataFrame()
        delta_mintues = X.apply(self._td_to_minutes)
        minute_pi_scale = 2*np.pi / 1440
        self.sin = np.sin(delta_mintues*minute_pi_scale)
        self.cos = np.cos(delta_mintues*minute_pi_scale)
        result[X.name+'_sin'] = self.sin
        result[X.name+'_cos'] = self.cos

        self.mean_sin, self.mean_cos = self._get_normed_mean_sin_cos(self.sin, self.cos)
        self.mean_time = self._sin_cos_to_time(self.mean_sin, self.mean_cos)

        return result

    def reverse(self, sin_col, cos_col):   
        pi_minute_scale = 1440 / (2*np.pi)             
        minutes = np.arctan2(sin_col, cos_col) * pi_minute_scale
        minutes[minutes < 0] = minutes[minutes <0] + ((2*np.pi) * pi_minute_scale)
        minutes = minutes.apply(lambda x: datetime(1,1,1, floor(x)//60, floor(x)%60).time())
        return minutes

    
    def _td_to_minutes(self, T):
        td = timedelta(hours=T.hour, minutes=T.minute)    
        return td.seconds/60

    def _get_normed_mean_sin_cos(self, sin_col, cos_col):
        mn_sin = sin_col.mean()
        mn_cos = cos_col.mean()
        norm = np.sqrt(mn_cos**2 + mn_sin**2)
        return mn_sin/norm, mn_cos/norm

    def _sin_cos_to_time(self, mean_sin, mean_cos):
        pi_minute_scale = 1440 / (2*np.pi)
        if np.arctan2(mean_sin, mean_cos) >= 0:
            minutes = int(np.arctan2(mean_sin, mean_cos) * pi_minute_scale) 
        elif np.arctan2(mean_sin ,mean_cos) < 0:
            minutes = int((np.arctan2(mean_sin, mean_cos) + (2*np.pi)) * pi_minute_scale) 
        return datetime(1,1,1,minutes//60, minutes%60).time()

        
class AvgRatioFiller:
    """
    "Fills NAs with appropriate values."
    
    INPUT:
        X_total: total sleep minutes column in float
        X_stage: stage sleep minutes column in float
    OUTPUT:
        X_stage: stage sleep minutes filled column in float
    """
    def __init__(self, X_total):
        self.X_total = X_total
    
    def fit(self, X_stage, y=None):
        self.X_stage = X_stage
        self.stage_ratio = self._get_stage_ratio(self.X_total, self.X_stage)
    
    def transform(self, X_stage):
        X_stage_transformed = X_stage.copy()
        X_stage_transformed.loc[X_stage.isna()] = self.X_total.loc[X_stage.isna()] * self.stage_ratio
        return X_stage_transformed
    
    def _get_stage_ratio(self, X_total, X_stage):
        mn_stage = X_stage.loc[~X_stage.isna()].mean()
        mn_total = X_total.loc[~X_stage.isna()].mean()
        return mn_stage/mn_total


class ZeroFiller:
    """
    "Fills NAs with zero values."
    
    INPUT:
        X_total: total sleep minutes column in float
        X_stage: stage sleep minutes column in float
    OUTPUT:
        X_stage: stage sleep minutes filled column in float
    """
    def __init__(self):
        pass
    
    def fit(self, series):
        self.series = series
    
    def transform(self, series):
        new_series = series.fillna(0).copy()
        return new_series


class MatrixPipeline:
    """
    "takes in an entire pd-df and transform each and every column as specified in PIPELINES"
    
    INPUT:
        branches: list of colname and transformer pairs in tuple
        matrix: a matrix in need of transformation in pd-df

    OUTPUT:
        new_matrix: a new matrix with selected features transformed in pd-df
    """

    def __init__(self, pipes):
        self.pipes = defaultdict(object)
        for target, pipeline in pipes:
            self.pipes[target] = pipeline

    def fit(self, df):
        self.df = df

    def transform(self, df):
        new_df = pd.DataFrame()
        for col in df.columns:
            if col in self.pipes.keys():
                pipeline = self.pipes[col]
                pipeline.fit(df[col])
                new_col = pipeline.transform(df[col])
                new_df = pd.concat([new_df, new_col], axis=1)
            else:
                new_df = pd.concat([new_df, df[col]], axis=1) 
        return new_df


class OneHotEncoder:
    """
    INPUT:
        series: a target column in pd-s

    OUTPUT:
        output_df: (a) transformed column(s) in pd-df
    """
    def __init__(self):
        pass

    def fit(self, series, y=None):
        self.X = series

    def transform(self, series):
        labels = series.unique()
        output_df = pd.DataFrame()
        for label in labels:
            output_df['is_'+str(label)] = (series == label) + 0

        return output_df 


class StandardScaler:
    def __init__(self):
        pass

    def fit(self, series, y=None):
        self.X = series

    def transform(self, series):
        mn = series.mean()
        std = series.std()
        new_series = (series - mn)/std
        return new_series


class ChainTransformer:
    def __init__(self, chain):
        self.chain = chain
    
    def fit(self, X, y=None):
        pass
    def transform(self, X):
        new_X = X.copy()
        for t in self.chain:
            t = t
            t.fit(new_X)
            new_X = t.transform(new_X)
        return new_X


class AvgFiller:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        new_X = X.copy()
        mn = new_X.mean()
        new_X.fillna(mn, inplace=True)
        return new_X



            
