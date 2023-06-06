# coding: utf-8
from model.preprocess.decompose.basicDecompose import Decomposer


class ManualWave(Decomposer):
    def __init__(self, data, params):
        super().__init__(data)
        self.n = params["n"]
        self.predict_len = params["predict_len"]

    def decompose(self):
        wave_data = self.data
        self.res = []
        for i in range(self.n):
            cA, cD = self.single_decompose(wave_data)
            wave_data = cA
            self.res.append(cD)
        self.res.append(wave_data)
        self.res = self.res[::-1]
        return self.res

    def single_decompose(self, wave_data):
        cA,cD = [],[]
        if len(wave_data) % 2:
            wave_data.append(wave_data[-1])
        for i in range(len(wave_data)//2):
            cA.append((wave_data[2 * i] + wave_data[2 * i + 1]) / 2 * 2 ** 0.5)
            cD.append((wave_data[2 * i] - wave_data[2 * i + 1]) / 2 * 2 ** 0.5)

        return cA,cD

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
            res = self.single_reconcrete(cA, cD)
            ca = res
        return ca[:self.predict_len]

    def single_reconcrete(self, cA, cD):
        assert len(cA) == len(cD),  "Not equal lens"
        res = []
        for i in range(len(cA)):
            cA1 = cA[i] + cD[i]
            cA2 = cA[i] - cD[i]
            res.append(cA1)
            res.append(cA2)
        return res
