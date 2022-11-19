import numpy as np
import pandas as pd
import math
import jieba
file_path ='cikuv2.log'
mylog = open('word.log', mode = 'w',encoding='utf-8')

def load_stop_words():
    """加载停用词"""
    with open("stop_words") as fr:
        stop_words = set([word.strip() for word in fr])
    return stop_words

with open(file_path, encoding="utf-8") as f:
    docs = f.readlines()
words = []
stop_words = load_stop_words()
for i in range(len(docs)):
    docs[i] = jieba.lcut(docs[i].strip("\n"))
    words += docs[i]
for i in words:
    if i not in stop_words:
        print(i,file=mylog)
#
# vocab = sorted(set(words), key=words.index)
# print(vocab,file=mylog)
