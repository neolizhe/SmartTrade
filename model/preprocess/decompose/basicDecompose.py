# coding:utf-8
# abstract  decompose method for time series
class Decomposer:
    def __init__(self, data, params=None):
        self.data = data
        self.params = params
        self.res = ""

    def decompose(self):
        pass

    def reconcrete(self, data_series):
        pass
