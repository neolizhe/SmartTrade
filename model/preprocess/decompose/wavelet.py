# coding:utf-8
# 2. pywt--dwt小波变换分解，cA低频趋势，cD1,cD2,cD3...多级高频分量
import pywt
from model.preprocess.decompose.basicDecompose import Decomposer


class Wavelet(Decomposer):
    def __init__(self, data, params):
        super().__init__(data)
        self.n = params["n"]
        self.predict_len = params["predict_len"]

    def decompose(self):
        wave_data = self.data
        self.res = []
        for i in range(self.n):
            cA, cD = pywt.dwt(wave_data, "haar")
            wave_data = cA
            self.res.append(cD)
        self.res.append(wave_data)
        self.res = self.res[::-1]
        return self.res

    def reconcrete(self, data_series):
        inverse_len = []
        lens = self.predict_len
        for data in data_series[:-1]:
            lens = lens // 2
            inverse_len.append(lens)
        inverse_len = inverse_len[::-1]
        ca = data_series[0]
        for index in range(1, len(data_series)):
            cd = data_series[index]
            cA = ca[:inverse_len[index - 1] + 1]
            cD = cd[:inverse_len[index - 1] + 1]
            res = pywt.idwt(cA, cD, "haar")
            ca = res
        return ca[:self.predict_len]
