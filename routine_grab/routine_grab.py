#!/bin/env SmartTrade
# coding:utf-8
from factor.comment.snowball.snowball import batchSnowball
from factor.comment.weibo.weibo import batchWeibo
import pandas as pds
import sys, os
import datetime
import logging
import logging.config as config

config.fileConfig('logging.conf')

if __name__ == "__main__":
    routine_date = datetime.date.today()
    stock_pool = ["贵州茅台", "宁德时代", "以岭药业", "金地集团", "金融街", "京东方"]
    total_comments = pds.DataFrame(columns=["create_time", "text", "stock_name"])
    logging.info("Start schedule date:%s, stocks to grab:%s" % (routine_date, "_".join(stock_pool)))
    for stock_name in stock_pool:
        snowball_comment_df = batchSnowball(key=stock_name, pages=100, interval=2)
        snowball_comment_df = snowball_comment_df.sort_values("create_time", ascending=True)
        logging.info("Snowball %s comments has been obtained! Size:%s" % (stock_name,
                                                                          len(snowball_comment_df)))
        # # weibo comment
        weibo_comment_df = batchWeibo(key=stock_name, pages=100, interval=2)
        weibo_comment_df = weibo_comment_df[["create_time", "text"]]
        logging.info("Weibo %s comments has been obtained! Size:%s" % (stock_name, len(weibo_comment_df)))
        #
        total_comment_df = pds.concat([snowball_comment_df, weibo_comment_df], axis=0, ignore_index=True)
        total_comment_df = total_comment_df.drop_duplicates(subset="text").reset_index(drop=True)
        total_comment_df.text = total_comment_df.text.map(lambda x: "_".join(x.split("\n")))
        total_comment_df["stock_name"] = stock_name

        if len(total_comments) < 1:
            total_comments = total_comment_df.copy()
        else:
            total_comments = pds.concat([total_comments, total_comment_df],
                                        axis=0, ignore_index=True)

    save_path = r"C:/Users/lizhe53/PycharmProjects/SmartTrade/data_pool/"
    print(total_comments)
    total_comments.to_csv(save_path + "stock_comments_{date}.csv".format(date=routine_date),
                          index=False)
    logging.info("Routine grab job done!")
