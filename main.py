# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json

from config.rootPath import root


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


from model.preprocess.decompose.wavelet import Wavelet
from model.preprocess.decompose.manualWave import ManualWave
import pandas as pds
import matplotlib.pyplot as plt
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pds.read_csv(r"C:\Users\lizhe53\PycharmProjects\SmartTrade\output\total_300750.SZ_data.csv",
                        encoding='utf-8')
    data = data.sort_values('time',ascending=True)
    y = data['ths_close_price_stock'].dropna().values

    wv = Wavelet(y,params={"n":5, "predict_len":10})
    res = wv.decompose()
    res = res[::-1]
    res.append(y)

    mmv = ManualWave(y,params={"n":5,"predict_len":10})
    res1 = mmv.decompose()
    res1 = res1[::-1]
    res1.append(y)
    # res = np.append(y, res, axis=0)
    [print(len(i)) for i in res]
    [print(len(i)) for i in res1]
    for i in range(len(res)):
        plt.subplot(len(res),1,i+1)
        plt.plot(res[i],label="wavelet")
        plt.plot(res1[i],label="manual")
    plt.show()
    plt.legend()

    import pywt
    cA, cD = pywt.dwt([1, 2, 3, 4,5], 'haar')
    print(cA/1.414, cD/1.414)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
