# coding:utf-8
import pandas as pds
from scipy.optimize import brute
from scipy.stats import normaltest
from statsmodels.tsa.arima_model import ARMA, ARIMA
from utils.common import pValue, blockPrint, enablePrint, rolling_mean, max_line, min_line


class Autoarima:
    def __init__(self, data, test_size=0.1):
        self.data = data
        self.test_rate = test_size
        self.model = ARMA(data, order=(1, 1))
        self._trainset_split()

    def _trainset_split(self):
        test_len = int(len(self.data) * self.test_rate)
        self.testset = self.data[-test_len:]
        self.trainset = self.data[:len(self.data) - test_len]

    def fit(self):

        def _objfunc(order, data, diff_i):
            fit = ARIMA(data, order=(order[0], diff_i, order[1])).fit()
            return fit.aic

        def _optimize_wrapper(data):
            # select best I
            # max order 2 for ARIMA model restriction
            i_order_set = [0, 1, 2]
            diff_i = 0
            data_pds = pds.Series(data)
            for i in i_order_set:
                if i == 0:
                    data_diff = data_pds
                else:
                    data_diff = data_pds.diff(i).dropna()
                if pValue(data_diff.values) <= 0.05:
                    diff_i = i
                    break
            grid = (slice(1, 3, 1), slice(1, 3, 1))
            blockPrint()
            res_order = brute(_objfunc, grid, args=(data, diff_i), finish=None, disp=False)
            return res_order, diff_i

        res_order, diff_i = _optimize_wrapper(self.data)
        if int(diff_i) == 0:
            best_order = [int(res_order[0]), int(res_order[1])]
            self.model = ARMA(self.data, order=best_order).fit(disp=-1)
        else:
            best_order = [int(res_order[0]), int(diff_i), int(res_order[1])]
            self.model = ARIMA(self.data, order=best_order).fit(disp=-1)
        enablePrint()
        print("best_order:%s" % best_order)
        print(normaltest(self.model.resid))

    def predict(self):
        print("start fit!")
        # Build Model, search best order
        iters = 0
        while iters < 3:
            try:
                self.fit()
                # Forecast
                prediction = self.model.forecast(len(self.testset), alpha=0.3)  # 95% conf
                return prediction
            except Exception as e:
                enablePrint()
                print(e)
                print("err:try again!")
                iters += 1

        fc = self.trainset[-len(self.testset):]
        fc = fc[::-1]
        m1 = rolling_mean(fc)
        m2 = max_line(fc)
        m3 = min_line(fc)
        conf = [[m3[index], m2[index]] for index in range(len(fc))]
        prediction = (m1, 0, conf)
        return prediction
