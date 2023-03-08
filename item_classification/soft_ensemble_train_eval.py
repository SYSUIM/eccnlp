# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from item_classification.utils import get_time_dif
import logging



# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        logging.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                labels_all, predict_all, loss_total, predict_pr_all = get_predict_label(config, model, dev_iter)
                dev_acc, dev_loss = evaluate(labels_all, predict_all, loss_total, len(dev_iter), test=False)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                logging.info(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logging.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    labels_all, predict_all,loss_total, predict_pr_all = get_predict_label(config, model, test_iter)
    evaluate(labels_all, predict_all, loss_total, len(test_iter), test=True)
    # test_acc, test_loss, test_report, test_confusion = evaluate(labels_all, predict_all, loss_total, len(test_iter), test=True)
    # msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    # logging.info(msg.format(test_loss, test_acc))
    # logging.info("Precision, Recall and F1-Score...")
    # logging.info(test_report)
    # logging.info("Confusion Matrix...")
    # logging.info(test_confusion)
    # time_dif = get_time_dif(start_time)
    # logging.info("Time usage:{}".format(time_dif))

def get_predict_label(config, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    predict_pr_all = np.array([], dtype=float)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predic_pr = torch.softmax(outputs.data, 1).cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            predict_pr_all = np.append(predict_pr_all, predic_pr)
    return labels_all, predict_all, loss_total, predict_pr_all


def evaluate(labels_all, predict_all, loss_total=1, data_iter_num=1, test=False):
    acc = metrics.accuracy_score(labels_all, predict_all)
    class_list = ['0', '1']
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        
        test_acc, test_report, test_confusion = acc, report, confusion
        # msg = 'Test Acc: {1:>6.2%}'
        # logging.info(msg.format(test_acc))
        logging.info("Precision, Recall and F1-Score...")
        logging.info(test_report)
        logging.info("Confusion Matrix...")
        logging.info(test_confusion)

        return acc, report, confusion
    return acc, loss_total / data_iter_num