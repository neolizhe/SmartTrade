# coding:utf-8
import logging
import time

import requests
import random
import pandas as pds
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import gc
from pyquery import PyQuery
from datetime import datetime

from config.rootPath import root


class basicSpyder:
    def __init__(self, url, params, request_engine='request', headers=None):
        self.url = url
        self.params = params
        self.request_engine = request_engine == 'selenium'
        self.headers = {
            "X-Forwarded-For": str(random.randint(0, 255)) + '.' + str(random.randint(0, 255)) + '.' + str(
                random.randint(0, 255)) + '.' + str(random.randint(0, 255)) + ":8080",
            #                 'X-Forwarded-For':'180.103.191.96:8080',
            "X-Requested-With": "XMLHttpRequest",
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:39.0) Gecko/20100101 Firefox/39.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive'}
        if headers:
            for key, value in headers.items():
                self.headers[key] = value
        self.session = 0
        self.request_engine_init()

    def request_engine_init(self):
        if self.request_engine:
            # webdriver.DesiredCapabilities.CHROME["proxy"] = {
            #     "httpProxy": self.headers['X-Forwarded-For'],
            #     "ftpProxy": self.headers['X-Forwarded-For'],
            #     "sslProxy": self.headers['X-Forwarded-For'],
            #     "proxyType": "MANUAL",
            # }
            # webdriver.DesiredCapabilities.CHROME["acceptSslCerts"] = True
            opt = webdriver.ChromeOptions()
            opt.add_argument("--headless")
            opt.add_argument("--no-sandbox")
            opt.add_argument("user-agent=" + self.headers["User-Agent"])
            driver_path = root("chromedriver.exe")
            print(driver_path)
            self.session = webdriver.Chrome(executable_path=driver_path, options=opt)
            self.session.header_overrides = self.headers
        else:
            self.session = requests.session()

    def single_request(self):
        if self.request_engine:
            self.session.get(self.url)
            input_box = WebDriverWait(self.session, 30).until(
                EC.visibility_of_element_located((By.XPATH, self.params["input_xpath"])))
            submit_button = WebDriverWait(self.session, 30).until(
                EC.element_to_be_clickable((By.XPATH, self.params["button_xpath"])))
            input_box.send_keys(self.params['q'])
            submit_button.click()
            print("跳转")
            cookie_list = self.session.get_cookies()
            cookie_str = ""
            for cookie_item in cookie_list:
                cookie_str += cookie_item["name"] + "=" + cookie_item["value"] + ";"
            self.cookies = cookie_str
            self.session.close()
        else:
            response = self.session.get(self.url, headers=self.headers, params=self.params, timeout=15)
            self.resp = response.json().copy()
            response.close()

    def start_request(self):
        trial_times = 0
        while trial_times < 3:
            try:
                self.single_request()
                break
            except Exception as e:
                trial_times += 1
                logging.warning("Err: %s detected in %s st request trail" % (e, trial_times))

    def get_cookies(self):
        pass

    def response_parser(self):
        pass

    def process(self):
        self.get_cookies()
        self.single_request()
        self.response_parser()

    def quit(self):
        if self.request_engine:
            self.session.quit()
        else:
            self.session.close()
        gc.collect()

if __name__ == "__main__":
    url = "https://www.114yygh.com/"
    params = {
        "input_xpath":"/html/body/div[1]/div[1]/div[1]/div/div[1]/div/div[1]/div[1]/input"
    }
    bs = basicSpyder(url=url,params=params,request_engine="selenium")
    bs.process()
