# coding:utf-8
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
import logging, warnings, sys, os


def time2dt(t):
    if type(t) == str:
        dt = datetime.strptime(t, "%a %b %d %H:%M:%S %z %Y")
    elif type(t) == int:
        if len(str(t)) > 10:
            times = len(str(t)) - 10
            t = t // 10 ** times
        dt = datetime.utcfromtimestamp(t)
    dtstr = dt.strftime("%Y-%m-%d")
    return dtstr


def delta_days(dt1, dt2):
    date1 = datetime.strptime(dt1, "%Y-%m-%d").date()
    date2 = datetime.strptime(dt2, "%Y-%m-%d").date()
    return (date1 - date2).days


def dt_after_days(dt, days=1):
    date1 = datetime.strptime(dt, "%Y-%m-%d").date()
    new_dt = date1 + timedelta(days=days)
    return datetime.strftime(datetime(new_dt.year, new_dt.month, new_dt.day), "%Y-%m-%d")


def pValue(df):
    res = adfuller(df)
    print("p-value:%f,lags used:%d" % (res[1], res[2]))
    return res[1]


def filter_nan(arr):
    return arr[~np.isnan(arr)]


# turn off scipy log message
def blockPrint():
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.FATAL)
    logging.getLogger('scipy').propagate = False
    sys._jupyter_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys._jupyter_stdout


# 计算auot_arima拟合失败情况下的均值线、最高、最低包络线
# 移动窗口平均值计算
def rolling_mean(data, window_size=12 * 4):
    res = []
    for index in range(len(data)):
        if index < window_size:
            res.append(np.mean(data[:window_size]))
        else:
            res.append(np.mean(data[index - window_size + 1: index + 1]))
    return res


# 上包络线
def max_line(data, window_size=12 * 4):
    res = [np.mean(data[:window_size])]
    for index in range(1, len(data)):
        v = data[index] if data[index] > res[-1] else res[-1]
        res.append(v)
    return res


# 下包络线
def min_line(data, window_size=12 * 4):
    res = [np.mean(data[:window_size])]
    for index in range(1, len(data)):
        v = data[index] if data[index] < res[-1] else res[-1]
        res.append(v)
    return res
