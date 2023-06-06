# coding:utf-8
from pyquery import PyQuery
import pandas as pds
from factor.comment.basciSpyder import basicSpyder
from urllib.request import unquote, quote
import time
from tqdm import tqdm

from utils.common import time2dt


def concat_url(url, params):
    url += quote("=1&q=", safe="/", encoding="utf=8")
    for k, v in params.items():
        if k == "q":
            url += quote(v, safe="/", encoding="utf-8")
        elif k == "page_type" or k == "page":
            k_v = "&" + k + "=" + v
            url += k_v
        else:
            pass
    return url


class Weibo(basicSpyder):
    def __init__(self, url, params):
        super().__init__(url, params)
        self.headers["X-XSRF-TOKEN"] = "15e394"
        self.headers["MWeibo-Pwa"] = "1"
        self.comment = pds.DataFrame()

    def response_parser(self):
        cards = self.resp.get('data').get('cards')
        try:
            for card_group in cards:
                if "card_group" not in card_group.keys():
                    continue
                for card in card_group["card_group"]:
                    if "mblog" not in card.keys():
                        continue
                    item = card.get('mblog')
                    weibo = {}
                    weibo['id'] = item.get('id')
                    weibo['text'] = PyQuery(item.get('text')).text()
                    weibo['attitudes'] = item.get('attitudes_count')
                    weibo['comments'] = item.get('comments_count')
                    weibo['reposts'] = item.get('reposts_count')
                    weibo['create_time'] = time2dt(item.get('created_at'))
                    self.comment = pds.concat([self.comment, pds.DataFrame(weibo, index=[0])], axis=0, ignore_index=True)
        except Exception as e:
            print(e)

def batchWeibo(key="", pages=1, interval=1.0):
    base_url = 'https://m.weibo.cn/api/container/getIndex?containerid=100103type'
    res_df = pds.DataFrame()
    wb = Weibo(url="url", params={})
    for page in tqdm(range(pages), desc="Grabbing Weibo comments..."):
        time.sleep(interval)
        params = {
            'containerid': "100103type=1",
            'q': key,
            'page_type': "search_all",
            'page': str(page + 1)
        }
        url = concat_url(base_url, params)
        wb.url = url
        wb.process()
        if len(wb.comment) < 1:
            break
        res_df = pds.concat([res_df, wb.comment], axis=0, ignore_index=True)

    wb.quit()
    return res_df


if __name__ == "__main__":
    res_df = batchWeibo("贵州茅台",2,interval=2.1)
    # print(res_df.columns)
    print(res_df[["text","create_time"]])
