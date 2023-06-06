# coding:utf-8
import time
from tqdm import tqdm
import pandas as pds
from pyquery import PyQuery
from factor.comment.basciSpyder import basicSpyder
from urllib.request import unquote, quote
from utils.common import time2dt


def get_cookies(key_word):
    # get cookies
    base_url = "http://xueqiu.com/k?q="
    input_xpath = "/html/body/div/nav/div/div[1]/form/input"
    button_xpath = "/html/body/div/nav/div/div[1]/form/span/button"
    params = {
        "q": key_word,
        "input_xpath": input_xpath,
        "button_xpath": button_xpath
    }
    cookie = basicSpyder(base_url, params=params, request_engine="selenium")
    cookie.process()
    res_cookie = cookie.cookies
    cookie.quit()
    return res_cookie


class Snowball(basicSpyder):
    def __init__(self, url, params):
        super().__init__(url, params)
        self.comment = pds.DataFrame()

    def response_parser(self):
        cards = self.resp.get("list")
        try:
            for card in cards:
                snowball = {}
                snowball["create_time"] = time2dt(card["created_at"])
                snowball["text"] = PyQuery(card["text"]).text()
                self.comment = pds.concat([self.comment, pds.DataFrame(snowball,
                                                                   index=[0])], axis=0, ignore_index=True)
        except Exception as e:
            print(e)


def batchSnowball(key="宁德时代", pages=1, cntPerPage=100, interval=1.0):
    # snowball comment
    base_url = "https://xueqiu.com/query/v1/search/status.json?sortId=1&q="
    origin_url = base_url + quote(key, safe="/", encoding="utf-8")
    cookies = get_cookies(key)

    res_df = pds.DataFrame()
    url = origin_url
    sn = Snowball(url=url, params={})
    sn.headers["Cookie"] = cookies

    for page in tqdm(range(pages),desc="Grabbing Snowball Comment..."):
        time.sleep(interval)
        url += "&count={cnt}&page={page}".format(cnt=cntPerPage, page=page)
        sn.url = url
        sn.process()
        if len(sn.comment) < 1:
            break
        res_df = pds.concat([res_df, sn.comment], axis=0, ignore_index=True)
        url = origin_url

    sn.quit()
    return res_df


if __name__ == "__main__":
    res_df = batchSnowball()
    print(res_df)
