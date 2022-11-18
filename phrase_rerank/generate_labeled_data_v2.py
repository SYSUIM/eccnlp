import functools
import difflib
import random
import numpy as np
# mylog1 = open('data_v5.txt', mode = 'w',encoding='utf-8')
mylog2 = open('labeled_v7.train', mode = 'w',encoding='utf-8')
mylog3 = open('labeled_v7.test', mode = 'w',encoding='utf-8')

#定义两个原因文本的相似度，返回值范围（0，1]
def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

#定义两个两个原因的排序标准
#第一维：相似度 1
#第二维：概率 2
#第三维：词出现的次数 3
def list_cmp(x,y):
    if x[1] != y[1]:
        return x[1]>y[1]
    elif x[2] != y[2]:
        return x[2]>y[2]
    else:
        return x[3]>y[3]

#读取词库中的词
vocab=[]
with open("word.log", "r", encoding="utf8") as f1:
    words = f1.readlines()
    for i in words:
        vocab.append(i.strip('\n'))


with open("my_ie1018v2.log", "r", encoding="utf8") as f:
    all_rows=[]  #
    pro_cnt=[]  #存词库中的词在原因中出现的次数
    key_num=-1  #标记quary
    lines = f.readlines()
    for i in range(len(lines)):
        if lines[i][0] != '{':
            continue
        data = eval(lines[i])
        if len(data) == 0:  # 预测为无因果
            continue
        elem_num=len(data["业绩归因"])
        # 预测出的因果不少于1个
        if elem_num>1:
            str2=lines[i-1][7:]
            data2=eval(str2)
            lab_txt=data2["result_list"][0]["text"] #当前quary下管院标记的原因
            #有因果标记
            if len(lab_txt) != 0:
                key_num+=1 #合法quary
                line_elem=[]
                for j in data["业绩归因"]:
                    elem=[]#当前quary 下每一个原因的特征向量
                    # 用预测的原因与标注原因的文本相似度作为相关度的值
                    elem.append(key_num)
                    tmp=string_similar(j["text"],lab_txt) #管院标记原因与UIE提取原因的相似度
                    #elem 每一维的含义
                    # 第0维：key_num  第一维：相似度  第二维：概率  第三维：词出现的次数
                    # 第四维：label（1，-1）  第5维：归一后的出现次数
                    elem.append(tmp)
                    elem.append(j["probability"])
                    #计算词库中的词在原因中出现的次数cot
                    cot = 0
                    for search_list in vocab:
                        if search_list in j["text"]:
                            cot=cot+1
                    elem.append(cot)
                    pro_cnt.append(cot)
                    line_elem.append(elem)
                # 排序，rank1标记label为1，其余为-1
                line_elem.sort(key=functools.cmp_to_key(list_cmp))
                for j in range(len(line_elem)):
                    if j==0:
                        line_elem[j].append(1)
                    else:
                        line_elem[j].append(-1)
                    all_rows.append(line_elem[j])
                    # print(line_elem[j])
#对词库中的词在原因中出现的次数cot归一化
cnt=[]
arr = np.asarray(pro_cnt)
for x in arr:
    x = float(x - np.min(arr))/(np.max(arr)- np.min(arr))
    cnt.append(x)
#构建input向量
all=[] #每个UIE提取原因的input格式
list_len=len(all_rows)
for i in range(list_len):
    line_in_all=[]
    line_in_all.append(all_rows[i][4]) #label
    line_in_all.append(all_rows[i][0]) #key_num
    line_in_all.append(all_rows[i][2]) #probability
    line_in_all.append(cnt[i])         #词库次数cot(归一化后）
    all.append(line_in_all)
#加入构建的负样本
alllist=[]#存所有正负样本
f_num=5
i=0
while i < len(all):
    alllist.append(all[i])
    t=1
    now_line=i
    while (all[now_line][1]==all[(now_line+t)%list_len][1]):
        t+=1
        i+=1
        alllist.append(all[i])
    #如果UIE提取的原因少于5个，随机生成缺少的负样本
    #负样本特征向量1范围（0，0.04），特征向量2范围（0，0.01）
    if f_num-t >0:
        for j in range(1,f_num-t+1):
            tmp=[]
            tmp.append(-1)
            tmp.append(all[now_line][1])
            tmp.append(random.random()*0.04)
            tmp.append(random.random()*0.01)
            alllist.append(tmp)
    i=i+1
#构建input格式  eg:  -1 qid:0 1:0.1 2:0.09
allline=[]
cott=-1
len_data=len(alllist)
for i in alllist:
    str1 = "" + str(i[0]) +" " +"qid:" + str(i[1])+" 1:"+str(i[2])+" 2:"+str(i[3])
    allline.append(str1)
    cott+=1
    if cott<len_data*0.7:
       print(str1,file=mylog2)
    if cott>=len_data*0.7:
        print(str1,file=mylog3)
