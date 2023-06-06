# coding : utf-8
'''
Greedy strategy day by day
'''
import pandas as pds
from model.model import *
from model.model.LSTMmodel import LSTMmodel
from model.model.TCNmodel import TCNmodel
from model.model.ConvNet import ConvNet
from model.model.WaveletPlusLSTM import WaveletPlusLSTM
from model.model.ConvPlusLSTM import ConvPlusLSTM
from model.model.NeuralProphet import MultiVariableNeuralProphet

if __name__ == "__main__":
    data = pds.read_csv(r"C:\Users\lizhe53\PycharmProjects\SmartTrade\output\total_300750.SZ_data.csv",
                        encoding='utf-8')
    data = data.sort_values("time", ascending=True).reset_index(drop=True)
    date_range = data.time.values
    data["label_copy"] = data.ths_close_price_stock
    data.drop(["Unnamed: 0", "time", "thscode"], axis=1, inplace=True)

    model_name = "MultiVariableNeuralProphet" # WaveletPlusLSTM>TCNmodel-> ConvNet -> ConvPlusLSTM
    tcn = eval(model_name)(df=data, date_range=date_range,
                  label_col='ths_close_price_stock',
                  window_size=128, epochs=10, batch_size=4, shuffle=False)
    tcn.data_process()
    tcn.build_model(input_shape=tcn.X_train.shape[1:])
    tcn.fit_process()
    tcn.predict_process()
    # predict on test set and give all y_hat in one time
    # [x1,x2,,,xn] -> [y1~,y2~,,,yn~]
    # Greedy Strategy
    # According to y_hat array , compute the Multi buy-sell points to Maximum gain.
    # According buy-sell points and real y array, compute the real gain.
    y_hat = list(tcn.model.predict(tcn.x_test))
    # Smooth prediction in given window size.Padding start.
    smooth_window = 7
    y_hat_smooth = []
    for index in range(len(y_hat)):
        if index < smooth_window:
            smooth_value = sum(y_hat[:index+1])/(index + 1)
        else:
            smooth_value = sum(y_hat[index-smooth_window:index+1])/smooth_window
        y_hat_smooth.append(smooth_value)
    y_hat_smooth.append(-1)
    # e.g y_hat = [1,2,3,2,4,5], best buy index = 0,3, best sell = 2,5
    # 考虑手续费带来的成本，买/卖0.2%
    buy_index_list, sell_index_list = [], []
    flag = -1

    for index in range(len(y_hat_smooth) - 1):
        if flag == -1 and y_hat_smooth[index] < y_hat_smooth[index + 1]:
            buy_index_list.append(index)

        elif flag == 1 and y_hat_smooth[index] > y_hat_smooth[index + 1]:
            if y_hat_smooth[index] > y_hat_smooth[buy_index_list[-1]] * 1.004:
                sell_index_list.append(index)
            else:
                buy_index_list.pop()

        if y_hat_smooth[index] <= y_hat_smooth[index + 1]:
            flag = 1
        else:
            flag = -1
    assert len(buy_index_list) == len(sell_index_list), "Buy size do not match sell size!"
    # Compute real gain
    total_gain = 0.0
    for index in range(len(buy_index_list)):
        buy_index = buy_index_list[index]
        sell_index = sell_index_list[index]
        total_gain += tcn.y_test[sell_index] * 0.998 - tcn.y_test[buy_index] * 1.002
        print("Buy index:%s, Sell index:%s" % (buy_index, sell_index))
        print("Predict Buy value:%s, sell value:%s" % (y_hat_smooth[buy_index], y_hat_smooth[sell_index]))
        print("Real Buy value:%s, sell value:%s" % (tcn.y_test[buy_index], tcn.y_test[sell_index]))

    print("Total gain for greedy strategy in test set:%s" % total_gain)
    roi_per_year = total_gain / tcn.y_test[buy_index_list[0]] * 365 / len(date_range) * 10 * 100
    print("Model:%s, Real ROI per year:%s %%" % (model_name, roi_per_year))
