# coding : utf - 8
# NeuralProphet model contains trend, seasonal, residual, holiday components
import pandas as pds
from neuralprophet import NeuralProphet, set_log_level
from model.model.LSTMmodel import LSTMmodel
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

set_log_level("WARNING")


class MultiVariableNeuralProphet(LSTMmodel):
    def __init__(self, df: pds.DataFrame, date_range: list, label_col="price",
                 window_size=7, test_size=0.1, shuffle=True, epochs=10, batch_size=4,
                 **kwargs):
        super().__init__(df, date_range, label_col,
                 window_size, test_size, shuffle, epochs, batch_size,
                 **kwargs)
        self.train_data_dict = {}
        self.test_data_dict = {}

    def data_process(self):
        super().data_process()
        train_date_range = self.date_range[:len(self.train_data)]
        for col in self.train_data.columns:
            if len(self.train_data[col].dropna().unique()) < 2:
                continue
            col_df = pds.DataFrame({'ds' : train_date_range, self.label_col : self.train_data[col].values})
            self.train_data_dict[col] = col_df

        # test data
        test_date_range = self.date_range[-len(self.test_data):]
        for col in self.train_data_dict.keys():
            col_df = pds.DataFrame({'ds' : test_date_range, self.label_col : self.test_data[col].values})
            self.test_data_dict[col] = col_df


    def build_model(self, input_shape):
        model = NeuralProphet(
            n_forecasts=1, # predict n steps
            n_lags=50, # AutoRegression laged steps to predict future
            #training parameters
            learning_rate=0.01,
            epochs=self.epochs, #train epochs
            batch_size=self.batch_size,
            loss_func="Huber",
            optimizer="AdamW",
            # Model structure parameters
            num_hidden_layers=6, # Hidden layers of  AR-Net and the FFNN of the lagged regressors
            d_hidden=256 # Hidden layer units
        )
        self.model = model

    def fit_process(self):
        self.loss = self.model.fit(self.train_data_dict, freq='D', validation_df=self.test_data_dict)
        # , progress="plot")
        print(self.loss.tail(1))

    def predict_process(self):
        choose_test_df_dict = dict((k, self.test_data_dict[k]) for k in [self.label_col])
        test_feature = self.model.make_future_dataframe(choose_test_df_dict, n_historic_predictions=True)
        y_pred = self.model.predict(test_feature)['y']
        loss_df = y_pred[['y','yhat1']].dropna()
        print("test set MSE:%s, MAPE:%s" % (mean_squared_error(loss_df['y'], loss_df['yhat1']),
                                            mean_absolute_percentage_error(loss_df['y'], loss_df['yhat1'])))
        # self.model.plot(y_pred[self.label_col])

        for col in y_pred.columns:
            if col == "ds":
                continue
            elif col == "y" or col == "yhat1":
                plt.subplot(211)
                plt.plot(y_pred['ds'], y_pred[col],label=col)
            else:
                plt.subplot(212)
                plt.plot(y_pred['ds'], y_pred[col], label=col)

        plt.legend()
        plt.show()



if __name__ == "__main__":

    data = pds.read_csv(r"C:\Users\lizhe53\PycharmProjects\SmartTrade\output\total_300750.SZ_data.csv",
                        encoding='utf-8')
    data = data.sort_values("time", ascending=True).reset_index(drop=True)
    date_range = data.time.values
    data.rename(columns={'ths_close_price_stock':'y'}, inplace=True)
    data.drop(["Unnamed: 0", "time", "thscode"], axis=1, inplace=True)
    model = MultiVariableNeuralProphet(df=data, date_range=date_range,
                        label_col='y',
                        window_size=128, epochs=2, batch_size=4, shuffle=False)
    model.process()