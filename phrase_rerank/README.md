lambdarank
Reference implementation of [Burges, C., et al., 2007]

模型参考https://github.com/Jabberwockleo

环境：tf2.2

1. python build_word.py

将管院标注的原因进行分词，储存在word.log中

2. python generate_labeled_data_v2.py

构建具有标记的正负样本数据，70%用于训练，30%用于预测

3. python train.py

训练模型

4. python predict_top1.py  预测 rank top1 的准确率

python predict_top2.py  预测 rank top2 的准确率

ps: 
mock.py:用于解析输入的文本数据并构建pair对

lambdarank.py :利用梯度上升法来更新参数
