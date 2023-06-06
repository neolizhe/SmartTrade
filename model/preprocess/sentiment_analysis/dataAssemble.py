# coding:utf-8
"""
1.stock info daily
2.weibo comment & snowball comment about target stock
!!! NOTICE: the command calls sentiment model must have the same directory as PyabsaModel.py to use offline
sentiment model checkoutpoint
"""
import logging
import os
import re
import pandas as pds
from factor.basic.iFindData.dataRequest import BatchTHS
from factor.comment.snowball.snowball import batchSnowball
from factor.comment.weibo.weibo import batchWeibo
from model.preprocess.sentiment_analysis.PyabsaModel import sentiment_classifier_batch
from datetime import datetime, timedelta


def get_dates_from_texts(raw_texts):
    expr = '\\[index\\][0-9\\-]{10}\\[index\\]'
    dates = []
    for text in raw_texts:
        res = re.search(expr, text)
        if res:
            index = res.span()
            dates.append(text[index[0] + 7: index[1] - 7])
        else:
            dates.append("1000-10-10")
    return dates


class dataAssemble:
    def __init__(self, stock_code, stock_name,
                 date_range="2017-12-31,2022-12-08",
                 output_path="output/", load_data_mode=False):
        self.stock_code = stock_code
        self.stock_name = stock_name
        self.date_range = date_range
        self.output_path = output_path
        self.final_data = pds.DataFrame()
        self.load_data_mode = load_data_mode
        self.grab_pages = 500

    def request_stock_info(self):
        # 最多可取5年前 且 总数据小于 1w
        bths = BatchTHS(self.stock_code, self.date_range)
        stock_info = bths.process()
        return stock_info

    def request_comment_info(self):
        # snowball comment request
        # columns [create_time, text]
        if self.load_data_mode:
            total_comment_df = pds.DataFrame()
            dir_path = r"C:\Users\lizhe53\PycharmProjects\SmartTrade\data_pool"
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                comments = pds.read_csv(file_path)
                comments = comments[comments.stock_name == self.stock_name]
                total_comment_df = pds.concat([total_comment_df, comments], axis=0, ignore_index=True)
        else:
            snowball_comment_df = batchSnowball(key=self.stock_name, pages=self.grab_pages, interval=1.1)
            snowball_comment_df = snowball_comment_df.sort_values("create_time", ascending=True)
            print("Snowball comments has been obtained! Size:%s" % len(snowball_comment_df))
            # weibo comment
            weibo_comment_df = batchWeibo(key=self.stock_name, pages=self.grab_pages, interval=1.2)
            weibo_comment_df = weibo_comment_df[["create_time", "text"]]
            print("Weibo comments has been obtained! Size:%s" % len(weibo_comment_df))
            total_comment_df = pds.concat([snowball_comment_df, weibo_comment_df], axis=0, ignore_index=True)

        total_comment_df = total_comment_df.drop_duplicates(subset="text").reset_index(drop=True)
        total_comment_df.text = total_comment_df.apply(lambda x: "[index]" + x.create_time + "[index]" +
                                                                 x.text, axis=1)
        scores, raw_texts = self.sentiment_aspect_analysis(total_comment_df.text.values)
        index_dates = get_dates_from_texts(raw_texts)
        aligned_size = min(len(index_dates), len(scores))
        total_score_df = pds.DataFrame({"create_time": index_dates[:aligned_size],
                                        "text": scores[:aligned_size]})
        group_col = "create_time"
        mean_scores = total_score_df.groupby(group_col).text.mean().reset_index(). \
            rename(columns={"text": "mean_score"})
        std_scores = total_score_df.groupby(group_col).text.std().reset_index(). \
            rename(columns={"text": "std_score"})
        quater_score = total_score_df.groupby(group_col).text.quantile(0.25).reset_index(). \
            rename(columns={"text": "quater_score"})
        quater3_score = total_score_df.groupby(group_col).text.quantile(0.75).reset_index(). \
            rename(columns={"text": "quater3_score"})

        total_score_df = mean_scores.join(std_scores.set_index(group_col), on=group_col, how="left") \
            .join(quater_score.set_index(group_col), on=group_col, how="left") \
            .join(quater3_score.set_index(group_col), on=group_col, how="left")

        return total_score_df

    def sentiment_aspect_analysis(self, sentences):
        # sentiment analysis
        asp_sentences = []
        asp_str = "[ASP]" + self.stock_name + "[ASP]"
        for s in sentences:
            raw_s = s.split(self.stock_name)
            asp_s = asp_str.join(raw_s)
            asp_sentences.append(asp_s)
        sentiment_list = sentiment_classifier_batch(asp_sentences)
        sentiment_scores, raw_texts = [], []
        for res in sentiment_list:
            final_score = 0.0
            raw_text = ""
            try:
                isPositive = True if res["sentiment"][0] == "Positive" else False
                score = float(res["confidence"][0])
                final_score = score if isPositive else -score
                raw_text = res["text"]
            except Exception as e:
                logging.warning("Err Detected as %s" % e)

            sentiment_scores.append(final_score)
            raw_texts.append(raw_text)

        return sentiment_scores, raw_texts

    def output_data(self):
        self.final_data.to_csv(self.output_path, encoding='utf-8', header=True)

    def main_process(self):
        stock_info_df = self.request_stock_info()
        comment_score_df = self.request_comment_info()
        self.final_data = stock_info_df.join(comment_score_df.set_index("create_time"), on="time", how="left")
        self.output_data()
        print("Data saved in:%s" % self.output_path)
        print(self.final_data.head())


if __name__ == "__main__":
    stock_code = "300750.SZ"
    stock_name = "宁德时代"
    date_range = "2017-12-01,2023-02-20"
    output_path = r"C:\Users\lizhe53\PycharmProjects\SmartTrade\output\total_{stock_code}_data.csv".format(
        stock_code=stock_code)
    data_model = dataAssemble(stock_code=stock_code, stock_name=stock_name,
                              date_range=date_range, output_path=output_path,
                              load_data_mode=True)
    data_model.main_process()
