#coding:utf-8
# 1. statsmodels使用的X-11分解过程，它主要将时序数据分离成长期趋势、季节趋势和随机成分

from statsmodels.tsa.seasonal import seasonal_decompose
from model.preprocess.decompose.basicDecompose import Decomposer


class Stats(Decomposer):
    def __init__(self, data, params):
        super().__init__(data)
        self.freq = params["freq"]

    def decompose(self):
        self.res = seasonal_decompose(self.data, freq=self.freq, model="additive")
        return [self.res.trend, self.res.seasonal, self.res.resid]

    def reconcrete(self, data_series):
        res = 0
        for data in data_series:
            res += data
        return res