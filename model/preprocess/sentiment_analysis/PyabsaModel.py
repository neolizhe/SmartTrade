# coding:utf-8
# -*- coding: utf-8 -*-
# file: extract_aspects_chinese.py
# github: https://github.com/yangheng95
import logging
import os

from findfile import find_file
from pyabsa.functional import ATEPCCheckpointManager, ABSADatasetList, available_checkpoints, APCCheckpointManager
from pyabsa.core.apc.prediction.sentiment_classifier import SentimentClassifier
from pyabsa.functional.checkpoint.checkpoint_manager import unzip_checkpoint
from torch.utils.data import DataLoader


class SentiClassifier(SentimentClassifier):
    def __init__(self, parent_model, **kwargs):
        checkpoint = "chinese"
        model_dir = r"C:\Users\lizhe53\PycharmProjects\SmartTrade\model\preprocess\sentiment_analysis\checkpoints"
        checkpoint_config = find_file(model_dir, [checkpoint, '.config'])
        checkpoint = os.path.dirname(checkpoint_config)

        super().__init__(model_arg=checkpoint, cal_perplexity=False, **kwargs)
        self.model = parent_model.model
        self.opt = parent_model.opt
        self.tokenizer = parent_model.tokenizer
        self.cal_perplexity = parent_model.cal_perplexity
        self.dataset = parent_model.dataset

    def batch_predict(self,
                      source_list,
                      print_result=True,
                      save_result=False,
                      ignore_error=True,
                      clear_input_samples=True):

        if clear_input_samples:
            self.clear_input_samples()

        save_path = os.path.join(os.getcwd(), 'apc_inference.result.json')
        samples = []
        for source in source_list:
            if source:
                samples.extend(self.dataset.parse_sample(source))
            else:
                logging.warning("No valid input sample")
        self.dataset.process_data(samples, ignore_error)
        self.infer_dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.eval_batch_size,
                                                 pin_memory=True,
                                                 shuffle=False)
        return self._infer(save_path=save_path if save_result else None, print_result=print_result)


# checkpoint_map = available_checkpoints(from_local=False)
def en_aspect_sentiment_analysis(sentence):
    source = []
    if not isinstance(sentence, list):
        source = [sentence]
    else:
        source = sentence

    inference_source = source
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='multilingual-256')
    atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source,
                                                   save_result=True,
                                                   print_result=True,  # print the result
                                                   pred_sentiment=True,
                                                   # Predict the sentiment of extracted aspect terms
                                                   )


def chinese_aspect_sentiment_analysis(sentence):
    source = []
    if not isinstance(sentence, list):
        source = [sentence]
    else:
        source = sentence

    # 从Google Drive下载提供的预训练模型
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='chinese', eval_batch_size=256)

    # examples = ABSADatasetList.Chinese
    res = aspect_extractor.extract_aspect(inference_source=source,  # list-support only, for now
                                          print_result=False,  # print the result
                                          pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                          )

    return res


def chinese_sentiment_classifier_single(sentence):
    if '[ASP]' not in sentence:
        sentence = '[ASP] Global Sentiment [ASP]' + sentence

    sentiment_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint="chinese", eval_batch_size=256)
    print(sentiment_classifier.infer(sentence, clear_input_samples=False))


def sentiment_classifier_batch(sentence_list):
    source = []
    if not isinstance(sentence_list, list):
        source = list(sentence_list)
    else:
        source = sentence_list

    asp_source = []
    for text in source:
        if '[ASP]' not in text:
            asp_source.append('[ASP] Global Sentiment [ASP]' + text)
        else:
            asp_source.append(text)

    sentiment_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint="chinese",
                                                                         eval_batch_size=256)
    sc_model = SentiClassifier(parent_model=sentiment_classifier)
    return sc_model.batch_predict(source_list=asp_source,print_result=False,ignore_error=False)


if __name__ == "__main__":
    examples1 = ['照 大 尺 寸 的 照 片 的 时 候 [ASP]手 机[ASP] 反 映 速 度 太 慢',
                 '关 键 的 时 候 需 要 表 现 持 续 影 像 的 短 片 功 能 还 是 很 有 用 的',
                 '相 比 较 原 系 列 [ASP]锐 度[ASP] 高 了 不 少 这 一 点 好 与 不 好 大 家 有 争 议',
                 '相 比 较 原 系 列 [ASP]锐 度[ASP] 高 了 不 少 这 一 点 好 与 不 好 大 家 有 争 议',
                 '相比较原系列锐度高了不少这一点好与不好大家有争议',
                 '相比较原系列锐度高了不少这一点好与不好大家有争议',
                 '这款手机的大小真的很薄，但是颜色不太好看， 总体上我很满意啦。']
    # res = chinese_aspect_sentiment_analysis(examples1)
    # res = sentiment_classifier_batch(examples1)
    import dataAssemble
    # da = dataAssemble.dataAssemble("1","2")
    # res = da.sentiment_aspect_analysis(examples1,)
    res = sentiment_classifier_batch(examples1)
    print(res,len(res))
