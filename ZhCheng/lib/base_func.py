import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
np.warnings.filterwarnings('ignore', 'Mean of empty slice')
import statsmodels.api as sm


min_period = None


def rank(df):
    return df.rank(axis=1, method='average', pct=True)


def delay(df, period=1):
    return df.shift(period)


def correlation(x, y, window=10, NaN_consist=True):
    results = x.rolling(window, min_period).corr(y)
    results.replace([np.inf, -np.inf], np.nan, inplace=True)
    if NaN_consist:
        results[x.isnull()] = np.nan
    return results


def covariance(x, y, window=10, NaN_consist=True):
    results = x.rolling(window, min_period).cov(y)
    if NaN_consist:
        results[x.isnull()] = np.nan
    return results


def scale(df, k=1):
    return df.mul(k).div(df.abs().sum(axis=1))


def delta(df, period=1):
    return df.diff(period)


def decay_linear(df, period=10, NaN_consist=True):
    results = df.copy()
    results.fillna(method='ffill', limit=3, inplace=True)
    df_val = results.values.copy()
    results_val = results.values
    weights = np.arange(period) + 1
    decay_weights = weights / weights.sum()
    for row in range(0, results.shape[0]-period+1):
        results_val[row+period-1] = decay_weights.dot(df_val[row : row+period])
    if NaN_consist:
        results[df.isnull()] = np.nan
    return results


def IndNeutralize(df, industry):
    if len(df.columns.tolist()) > len(industry.columns.tolist()):
        df = df[industry.columns]
    else:
        industry = industry[df.columns]
    df_mean = pd.DataFrame(index = df.index, columns = industry.index)
    df_std = pd.DataFrame(index = df.index, columns = industry.index)
    for i in industry.index.tolist():
        x = industry.loc[i].dropna()
        a = df[x.index.tolist()]
        a = a.fillna(np.inf)
        aa = a[abs(a) != np.inf].mean(axis = 1)
        b = a[abs(a) != np.inf].std(axis = 1)
        df_mean[i] = aa
        df_std[i] = b
    mean_m = np.matrix(df_mean.fillna(0))
    #std_m = np.matrix(df_std.fillna(1))
    industry_m = np.matrix(industry.fillna(0))
    mean_m = mean_m * industry_m
    #std_m = std_m * industry_m
    df_m = np.matrix(df)
    neutralize_m = (df_m - mean_m)
    return pd.DataFrame(neutralize_m, index = df.index, columns = df.columns)


def industry_neutralize(df, industry):
    results = df.copy()
    results.columns = list(range(results.shape[1]))
    results_val = results.values
    for day in range(df.shape[0]):
        ind_groups = results.iloc[day].groupby(industry.iloc[day].values).groups
        for ind in ind_groups:
            cols = ind_groups[ind]
            results_val[day, cols] -= np.nanmean(results_val[day, cols])
    results.columns = df.columns
    return results


def ts_min(df, window=10, NaN_consist=True):
    results = df.rolling(window, min_period).min()
    if NaN_consist:
        results[df.isnull()] = np.nan
    return results


def ts_max(df, window=10, NaN_consist=True):
    results = df.rolling(window, min_period).max()
    if NaN_consist:
        results[df.isnull()] = np.nan
    return results


def ts_argmax(df, window=10, NaN_consist=True):
    results = window - df.rolling(window, min_period).apply(np.nanargmax)
    if NaN_consist:
        results[df.isnull()] = np.nan
    return results


def ts_argmin(df, window=10, NaN_consist=True):
    results = window - df.rolling(window, min_period).apply(np.nanargmin)
    if NaN_consist:
        results[df.isnull()] = np.nan
    return results


def ts_rank(df, window=10):
    rank_results = []
    for i in range(window-1, len(df)):
        temp = df.iloc[i-window+1 : i+1].rank(method='average', pct=True)
        rank_results.append(temp.iloc[-1].copy())
    rank_results = pd.concat(rank_results, axis=1).T
    rank_NA = pd.DataFrame(index=df.index[:window-1], columns=df.columns)
    rank_results = pd.concat([rank_NA, rank_results])
    return rank_results


def ts_sum(df, window=10, NaN_consist=True):
    results = df.rolling(window, min_period).sum()
    if NaN_consist:
        results[df.isnull()] = np.nan
    return results


def product(df, window=10, NaN_consist=True):
    results = df.rolling(window, min_period).apply(np.nanprod)
    if NaN_consist:
        results[df.isnull()] = np.nan
    return results


def stddev(df, window=10, NaN_consist=True):
    results = df.rolling(window, min_period).std()
    if NaN_consist:
        results[df.isnull()] = np.nan
    return results


def ts_mean(df, window=10, NaN_consist=True):
    results = df.rolling(window, min_period).mean()
    if NaN_consist:
        results[df.isnull()] = np.nan
    return results


def to_zscore(df, axis=1):
    return df.transform(lambda x: (x-x.mean()) / x.std(), axis=axis)


def winsorize(df):
    result = df.copy()
    mean_ = df.mean(axis=1)
    std_ = df.std(axis=1)
    left_3std =  mean_ - 3*std_
    right_3std = mean_ + 3*std_
    
    normal_bool = ((df.T >= left_3std) & (df.T <= right_3std)).T
    normal = df[normal_bool]
    normal_min = normal.min(axis=1)
    normal_max = normal.max(axis=1)
    
    normal_min = pd.DataFrame([normal_min]*df.shape[1], index=df.columns).T
    normal_max = pd.DataFrame([normal_max]*df.shape[1], index=df.columns).T
    result[(df.T < left_3std).T] = normal_min
    result[(df.T > right_3std).T] = normal_max
    return result


def to_0_1(df):
    return df.transform(lambda x:(x-x.min()) / (x.max()-x.min()), 
                        axis=1)


def outliers_revise(df): 
    mc = pd.Series(index=df.index)
    for day in df.index:
        day_data = df.loc[day].dropna()
        if len(day_data) > 0:
            mc.loc[day] = sm.stats.stattools.medcouple(day_data)

    Q1 = df.quantile(0.25, axis=1)
    Q3 = df.quantile(0.75, axis=1)
    IQR = Q3 - Q1

    L = pd.Series(index=df.index)
    L_pos = Q1 - 1.5 * np.exp(-3.5*mc) * IQR
    L[mc >= 0] = L_pos
    L_neg = Q1 - 1.5 * np.exp(-4*mc) * IQR
    L[mc < 0] = L_neg

    U = pd.Series(index=df.index)
    U_pos = Q3 + 1.5 * np.exp(4*mc) * IQR
    U[mc >= 0] = U_pos
    U_neg = Q3 + 1.5 * np.exp(3.5*mc) * IQR
    U[mc < 0] = U_neg

    normal_bool = ((df.T >= L) & (df.T <= U)).T
    normal_df = df[normal_bool]
    normal_min = normal_df.min(axis=1)
    normal_max = normal_df.max(axis=1)
    
    result = df.copy()
    normal_min = pd.DataFrame([normal_min]*df.shape[1], index=df.columns).T
    normal_max = pd.DataFrame([normal_max]*df.shape[1], index=df.columns).T    
    result[(df.T < L).T] = normal_min
    result[(df.T > U).T] = normal_max
    
    return result

trade_data_path = None
def read_trade_data(*args, data_path=None):
    if data_path is None:
        data_path = trade_data_path        
    with pd.HDFStore(data_path) as trade_data:
        data = []
        for d in args:
            data.append(trade_data[d])
    return tuple(data)