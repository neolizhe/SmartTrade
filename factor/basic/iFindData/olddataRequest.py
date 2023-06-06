# coding:utf-8
# get stock history basic info from iFind trail interface
import requests
from iFinDPy import *
import pandas as pds
from threading import Thread, Lock, Semaphore
import functools
from datetime import datetime
import json
from config.rootPath import root
from output.outputPath import output
from utils.common import delta_days, dt_after_days

MAX_DATA_SIZE = 10000  # 试用版限制
MAX_DELTA_DAYS = 360

def only_run_once(func):
    @functools.wraps(func)
    def wrapper_only_run_once(*args, **kwargs):
        if wrapper_only_run_once.first_call == 0:
            return wrapper_only_run_once.result
        else:
            wrapper_only_run_once.first_call = 0
            wrapper_only_run_once.result = func(*args, **kwargs)
            return wrapper_only_run_once.result

    wrapper_only_run_once.first_call = 1
    wrapper_only_run_once.result = 0
    return wrapper_only_run_once


# 登录函数
@only_run_once
def login():
    # 输入用户的帐号和密码
    users = pds.read_csv("../../../config/iFindUsers.csv")
    users.columns = ["user", "pass", "info"]
    index = 0
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


class iFind:
    def __init__(self, stock_code="", date_range=None):
        self.sem = Semaphore(5)  # 此变量用于控制最大并发数
        self.dllock = Lock()  # 此变量用来控制实时行情推送中落数据到本地的锁
        self.stockCode = stock_code
        self.dateRange = date_range
        login()

    def datepool_basicdata_demo(self):
        # 通过数据池的板块成分函数和基础数据函数，提取沪深300的全部股票在2020-11-16日的日不复权收盘价
        data_hs300 = THS_DP('block', '2020-11-16;001005290', 'date:Y,thscode:Y,security_name:Y')
        if data_hs300.errorcode != 0:
            print('error:{}'.format(data_hs300.errmsg))
        else:
            seccode_hs300_list = data_hs300.data['THSCODE'].tolist()
            data_result = THS_BD(seccode_hs300_list, 'ths_close_price_stock', '2020-11-16,100')
            if data_result.errorcode != 0:
                print('error:{}'.format(data_result.errmsg))
            else:
                data_df = data_result.data
                print(data_df)

    def datapool_realtime_demo(self):
        # 通过数据池的板块成分函数和实时行情函数，提取上证50的全部股票的最新价数据,并将其导出为csv文件
        today_str = datetime.today().strftime('%Y-%m-%d')
        print('today:{}'.format(today_str))
        data_sz50 = THS_DP('block', '{};001005260'.format(today_str), 'date:Y,thscode:Y,security_name:Y')
        if data_sz50.errorcode != 0:
            print('error:{}'.format(data_sz50.errmsg))
        else:
            seccode_sz50_list = data_sz50.data['THSCODE'].tolist()
            data_result = THS_RQ(seccode_sz50_list, 'latest')
            if data_result.errorcode != 0:
                print('error:{}'.format(data_result.errmsg))
            else:
                data_df = data_result.data
                print(data_df)
                data_df.to_csv('realtimedata_{}.csv'.format(today_str))

    def iwencai_demo(self):
        # 演示如何通过不消耗流量的自然语言语句调用常用数据
        print('输出资金流向数据')
        data_wencai_zjlx = THS_WC('主力资金流向', 'stock')
        if data_wencai_zjlx.errorcode != 0:
            print('error:{}'.format(data_wencai_zjlx.errmsg))
        else:
            print(data_wencai_zjlx.data)

        print('输出股性评分数据')
        data_wencai_xny = THS_WC('股性评分', 'stock')
        if data_wencai_xny.errorcode != 0:
            print('error:{}'.format(data_wencai_xny.errmsg))
        else:
            print(data_wencai_xny.data)

    def dlwork(self, tick_data):
        # 本函数为实时行情订阅新启线程的任务函数
        self.dllock.acquire()
        with open('dlwork.txt', 'a') as f:
            for stock_data in tick_data['tables']:
                if 'time' in stock_data:
                    timestr = _time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(stock_data['time'][0]))
                    print(timestr)
                    f.write(timestr + str(stock_data) + '\n')
                else:
                    pass
        self.dllock.release()

    def work(self, codestr, lock, indilist):
        self.sem.acquire()
        stockdata = THS_HF(codestr, ';'.join(indilist), '', '2020-08-11 09:15:00', '2020-08-11 15:30:00', 'format:json')
        if stockdata.errorcode != 0:
            print('error:{}'.format(stockdata.errmsg))
            self.sem.release()
        else:
            print(stockdata.data)
            lock.acquire()
            with open('test1.txt', 'a') as f:
                f.write(str(stockdata.data) + '\n')
            lock.release()
            self.sem.release()

    def multiThread_demo(self):
        # 本函数为通过高频序列函数,演示如何使用多线程加速数据提取的示例,本例中通过将所有A股分100组,最大线程数量sem进行提取
        # 用户可以根据自身场景进行修改
        today_str = datetime.today().strftime('%Y-%m-%d')
        print('today:{}'.format(today_str))
        data_alla = THS_DP('block', '{};001005010'.format(today_str), 'date:Y,thscode:Y,security_name:Y')
        if data_alla.errorcode != 0:
            print('error:{}'.format(data_alla.errmsg))
        else:
            stock_list = data_alla.data['THSCODE'].tolist()

        indi_list = ['close', 'high', 'low', 'volume']
        lock = Lock()

        btime = datetime.now()
        l = []
        for eachlist in [stock_list[i:i + int(len(stock_list) / 10)] for i in
                         range(0, len(stock_list), int(len(stock_list) / 10))]:
            nowstr = ','.join(eachlist)
            p = Thread(target=self.work, args=(nowstr, lock, indi_list))
            l.append(p)

        for p in l:
            p.start()
        for p in l:
            p.join()
        etime = datetime.now()
        print(etime - btime)

    def reportDownload(self):
        df = THS_ReportQuery('300033.SZ', 'beginrDate:2021-08-01;endrDate:2021-08-31;reportType:901',
                             'reportDate:Y,thscode:Y,secName:Y,ctime:Y,reportTitle:Y,pdfURL:Y,seq:Y').data
        print(df)
        for i in range(len(df)):
            pdfName = df.iloc[i, 4] + str(df.iloc[i, 6]) + '.pdf'
            pdfURL = df.iloc[i, 5]
            r = requests.get(pdfURL)
            with open(pdfName, 'wb+') as f:
                f.write(r.content)

    def basicStockInfo(self):
        start_date, end_date = self.dateRange.split(",")
        resp = THS_HF(self.stockCode, 'open;high;low;close',
                      'CPS:no,baseDate:1900-01-01,Fill:Previous,Interval:5,Days:WorkDays',
                      start_date + ' ' + '09:30:00', end_date + ' ' + '09:40:00')
        return resp.data

    def process(self):
        # 本脚本为数据接口通用场景的实例,可以通过取消注释下列示例函数来观察效果

        # 通过数据池的板块成分函数和基础数据函数，提取沪深300的全部股票在2020-11-16日的日不复权收盘价
        # self.datepool_basicdata_demo()
        # 通过数据池的板块成分函数和实时行情函数，提取上证50的全部股票的最新价数据,并将其导出为csv文件
        # self.datapool_realtime_demo()
        # 演示如何通过不消耗流量的自然语言语句调用常用数据
        # iwencai_demo()
        # 本函数为通过高频序列函数,演示如何使用多线程加速数据提取的示例,本例中通过将所有A股分100组,最大线程数量sem进行提取
        # multiThread_demo()
        # 本函数演示如何使用公告函数提取满足条件的公告，并下载其pdf
        # reportDownload()
        # high freq history 5min as unit
        return self.basicStockInfo()


def iFindProcess(stock_code="300750.SZ", date_range="2022-08-01,2022-09-01"):
    # stock_code = "300750.SZ"
    # start = "2022-08-01"
    # end = "2022-09-01"
    st = iFind(stock_code, date_range=date_range)
    df = st.process()
    return df


def read_stock_book(stock_code="300750.SZ", date_range="2022-09-04,2022-10-04",json_file="StockInfoBook.json"):
    start_dt, end_dt = date_range.split(",")
    global_param = 'Days:WorkDays,Fill:Previous,Interval:D'

    json_path = root(json_file)
    with open(json_path, "r", encoding="UTF-8") as js:
        stock_dict = json.load(js)

    batch_request_dict = {}
    for key, value in stock_dict.items():
        if key[0] == "_":
            continue
        batch_key = value[1]
        if batch_key in batch_request_dict.keys():
            batch_request_dict[batch_key] += ";" + value[0]
        else:
            batch_request_dict[batch_key] = value[0]

    # output dataframe
    df = pds.DataFrame()
    for key, value in batch_request_dict.items():
        print(key,value)
        param_set = key.split(",")
        short_param = ""
        if len(param_set) > 1:
            short_param = ",".join(param_set[1:])
        else:
            short_param = param_set[0]
        words = value.split(";")
        request_param = ""
        for i in range(len(words)):
            if i == 0:
                request_param += short_param
            else:
                request_param += ";" + short_param
        max_days = min(MAX_DATA_SIZE // len(words), MAX_DELTA_DAYS)
        iter_start = start_dt
        res_df = pds.DataFrame()
        while True:
            left_days = delta_days(end_dt, iter_start)
            if left_days <= 0:
                break
            if max_days > left_days:
                iter_end = end_dt
            else:
                iter_end = dt_after_days(iter_start, max_days)
            iter_df = THS_DS(stock_code, value, request_param, global_param, iter_start, iter_end)

            if len(res_df) < 1:
                res_df = iter_df.data
            else:
                res_df = pds.concat([res_df, iter_df.data], axis=0, ignore_index=True)
            iter_start = dt_after_days(iter_end, 1)

        # print(res_df.head(), request_param)
        if len(df) < 1:
            df = res_df
        else:
            df = df.join(res_df.set_index(["time", "thscode"]), on=["time", "thscode"], how="left")
    return df


if __name__ == "__main__":
    res = iFindProcess()
    # 基本情况
    stock_code = "300750.SZ"
    start = '2022-08-01 09:30:00'
    end = '2022-09-01 09:30:00'
    # bd_key
    print("".split(","))
    # df = read_stock_book(stock_code,"2021-10-01,2022-09-20")
    # df.to_csv(output("ths_data.csv"), encoding='gbk')
