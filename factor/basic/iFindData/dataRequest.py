# coding:utf-8
# get stock history basic info from iFind trail interface
from iFinDPy import *
import pandas as pds
import numpy as np
import json
from config.rootPath import root
from output.outputPath import output
from utils.common import delta_days, dt_after_days

MAX_DATA_SIZE = 20000  # 试用版限制
MAX_DELTA_DAYS = 360
USER_INDEX = 0


class BatchTHS:
    def __init__(self, stock_code, date_range):
        self.stock_code = stock_code
        self.global_param = 'Days:WorkDays,Fill:Previous,Interval:D'
        self.start_dt, self.end_dt = date_range.split(",")
        self.batch_request_dict = {}
        self._login()

    # 登录函数
    @staticmethod
    def _login():
        # 输入用户的帐号和密码
        users = pds.read_csv(r"C:\Users\lizhe53\PycharmProjects\SmartTrade\config\iFindUsers.csv")
        users.columns = ["user", "pass", "info"]
        index = USER_INDEX
        while True:
            if index >= len(users):
                raise Exception("Err: no valid user!")
            user, password = users.loc[index, "user"], users.loc[index, "pass"]
            print(user, password)
            thsLogin = THS_iFinDLogin(user, str(password))
            if thsLogin != 0:
                index += 1
            else:
                print('登录成功')
                break

    @staticmethod
    def _logout():
        THS_iFinDLogout()

    def high_freq_info(self):
        resp = THS_HF(self.stock_code, 'open;high;low;close',
                      'CPS:no,baseDate:1900-01-01,Fill:Previous,Interval:5,Days:WorkDays',
                      self.start_dt + ' ' + '09:30:00', self.end_dt + ' ' + '09:40:00')
        return resp.data

    def assemble_request(self):
        json_path = root("StockInfoBook.json")
        with open(json_path, "r", encoding="UTF-8") as js:
            stock_dict = json.load(js)

        batch_request_dict = {}
        for key, value in stock_dict.items():
            if key[0] == "_":
                continue
            batch_key = value[1]
            if batch_key in batch_request_dict.keys():
                batch_request_dict[batch_key].append(value[0])
            else:
                batch_request_dict[batch_key] = [value[0]]
        self.batch_request_dict = batch_request_dict

    # time unit is Year or None
    def batch_request(self):
        static_df = pds.DataFrame()
        dynamic_df = static_df.copy(deep=True)
        year_list = range(int(self.start_dt[:4]), int(self.end_dt[:4]) + 1)

        for key, value in self.batch_request_dict.items():
            indicator = ";".join(value)
            if key == "2022":
                year_df = pds.DataFrame()
                for request_year in year_list:
                    year_param = ";".join([str(request_year)] * len(value))
                    year_res = THS_BD(self.stock_code, indicator, year_param)
                    year_res.data["time"] = str(request_year)
                    year_df = pds.concat([year_df, year_res.data], axis=0, ignore_index=True)
                if len(static_df) < 1:
                    static_df = year_df
                else:
                    static_df = static_df.join(year_df.set_index("thscode"), on="thscode", how="left")
            elif key == "":
                notime_param = ";" * (len(value) - 1)
                notime_df = THS_BD(self.stock_code, indicator, notime_param).data
                if len(static_df) < 1:
                    static_df = notime_df
                else:
                    static_df = static_df.join(notime_df.set_index("thscode"), on="thscode", how="left")
            else:
                dynamic_res = self.dynamic_request(key, value)
                if len(dynamic_df) < 1:
                    dynamic_df = dynamic_res
                else:
                    dynamic_df = dynamic_df.join(dynamic_res.set_index(["thscode", "time"]),
                                                 on=["thscode", "time"], how="left", rsuffix="_r")
        dynamic_df["year"] = dynamic_df.time.map(lambda x: x[:4])
        tol_df = dynamic_df.join(static_df.set_index(["thscode", "time"]),
                                 on=["thscode", "year"], how="left", rsuffix="_static")
        return tol_df

    # time unit is Day
    def dynamic_request(self, key, value):
        indicator = ";".join(value)
        param_set = key.split(",")
        short_param = ""
        if len(param_set) > 1:
            short_param = ",".join(param_set[1:])
        else:
            short_param = param_set[0]

        request_param = ";".join([short_param] * len(value))
        max_days = min(MAX_DATA_SIZE // len(value), MAX_DELTA_DAYS)
        iter_start = self.start_dt
        res_df = pds.DataFrame()
        while True:
            left_days = delta_days(self.end_dt, iter_start)
            if left_days <= 0:
                break
            if max_days > left_days:
                iter_end = self.end_dt
            else:
                iter_end = dt_after_days(iter_start, max_days)
            iter_res = THS_DS(self.stock_code, indicator, request_param, self.global_param, iter_start, iter_end)
            if iter_res.errorcode != 0:
                self._logout()
                global USER_INDEX
                USER_INDEX += 1
                self._login()
                continue
            else:
                pass
            if len(res_df) < 1:
                res_df = iter_res.data
            else:
                res_df = pds.concat([res_df, iter_res.data], axis=0, ignore_index=True)
            iter_start = dt_after_days(iter_end, 1)
        return res_df

    def process(self):
        self.assemble_request()
        return self.batch_request()


if __name__ == "__main__":
    # res = THS_iFinDLogin("lydx036","828877")
    # print(res)

    # res = iFindProcess()
    # 基本情况
    stock_code = "300750.SZ"
    end_dt = "2022-10-05"
    #最多可取5年前 且 总数据小于 1w
    start_time = datetime.now().date() + timedelta(days=-365*5 + 1)
    start_dt = datetime.strftime(datetime(start_time.year, start_time.month, start_time.day),"%Y-%m-%d")
    print(start_dt)
    bths = BatchTHS(stock_code, "%s,%s" % (start_dt,end_dt))
    res_df = bths.process()
    res_df.to_csv(output("ths_data.csv"), encoding='gbk')

