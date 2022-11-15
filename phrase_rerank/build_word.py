import numpy as np
import pandas as pd
import math
import jieba
file_path ='cikuv2.log'
mylog = open('word.log', mode = 'w',encoding='utf-8')
with open(file_path, encoding="utf-8") as f:
    docs = f.readlines()
words = []
for i in range(len(docs)):
    docs[i] = jieba.lcut(docs[i].strip("\n"))
    words += docs[i]
for i in words:
    print(i,file=mylog)
#
# vocab = sorted(set(words), key=words.index)
# print(vocab,file=mylog)