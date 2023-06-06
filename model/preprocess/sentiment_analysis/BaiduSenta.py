# coding:utf-8
# 百度Senta开源情感预测接口
import numpy as np
from senta import Senta
from senta.common.rule import InstanceName
from senta.data.util_helper import convert_texts_to_ids, structure_fields_dict
from senta.utils.util_helper import array2tensor, text_type


# init model saved path
# C:\Users\lizhe53\AppData\Local\Continuum\anaconda3\envs\SmartTrade\Lib\site-packages\senta

class SentimentChineseClassifier(Senta):
    def __init__(self, task="sentiment_classify"):
        super(SentimentChineseClassifier, self).__init__()
        self.init_model(task=task)

    def predict(self, texts_, aspects=None):
        if isinstance(texts_, text_type):
            texts_ = [texts_]

        if isinstance(aspects, text_type):
            aspects = [aspects]

        return_list = convert_texts_to_ids(
            texts_, self.tokenizer, self.max_seq_len, self.truncation_type, self.padding_id)
        record_dict = structure_fields_dict(return_list, 0, need_emb=False)
        input_list = []
        for item in self.input_keys:
            kv = item.split("#")
            name = kv[0]
            key = kv[1]
            input_item = record_dict[InstanceName.RECORD_ID][key]
            input_list.append(input_item)
        inputs = [array2tensor(ndarray) for ndarray in input_list]
        result = self.inference.run(inputs)
        batch_result = self.model_class.parse_predict_result(result)
        results = []
        if self.inference_type == 'seq_lab':
            for text, probs in zip(texts_, batch_result):
                label = [self.label_map[l] for l in probs]
                results.append((text, label, probs.tolist()))
        else:
            for text, probs in zip(texts_, batch_result):
                label = self.label_map[np.argmax(probs)]
                results.append((text, label, probs.tolist()))
        return results


if __name__ == "__main__":
    classifier = SentimentChineseClassifier(task="aspect_sentiment_classify")
    texts = ["PHP 是世界上最好的语言"]
    aspects = ["世界"]
    result = classifier.predict(texts,aspects)
    print(result)
    # scikit-learn              0.20.4
    my_senta = Senta()

    # 获取目前支持的情感预训练模型, 我们开放了以ERNIE 1.0 large(中文)、ERNIE 2.0 large(英文)和RoBERTa large(英文)作为初始化的SKEP模型
    print(
        my_senta.get_support_model())  # ["ernie_1.0_skep_large_ch", "ernie_2.0_skep_large_en", "roberta_skep_large_en"]

    # 获取目前支持的预测任务
    print(my_senta.get_support_task())  # ["sentiment_classify", "aspect_sentiment_classify", "extraction"]


    # 预测中文句子级情感分类任务
    # my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="sentiment_classify", use_cuda=use_cuda)
    # texts = ["贵州今天走势不好"]
    # result = my_senta.predict(texts)
    # print(result)
    #
    # # 预测中文评价对象级的情感分类任务
    my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="aspect_sentiment_classify", use_cuda=False)
    texts = ["贵州今天走势不好"]
    aspects = ["贵州"]
    result = my_senta.predict(texts, aspects)
    print(result)
