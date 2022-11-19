#!/usr/bin/env python
# -*- coding: utf-8 -*-

NO_LAMBDA_MEASURE_USING_SGD = "pure_sgd"
LAMBDA_MEASURE_AUC = "factorized_pairwise_precision"
LAMBDA_MEASURE_NDCG = "normalized_discounted_cumulative_gain"
#LAMBDA_MEASURE_MRR = "mean_reciprocal_rank"
#LAMBDA_MEASURE_MAP = "mean_average_precision"
#LAMBDA_MEASURE_ERR = "expected_reciprocal_rank"

DEBUG_LOG = True
QUALITY_MEASURE = LAMBDA_MEASURE_NDCG
USE_HIDDEN_LAYER = True
USE_TOY_DATA = False
# USE_TOY_DATA = True
LAYER_WIDTH = 10
FEATURE_NUM = 2
LEARNING_RATE = 0.001
MODEL_PATH = "./data_model_v7_lambdarank.ckpt"
TRAIN_DATA = "./labeled_v7.train"
TEST_DATA = "./labeled_v7.test"
LOSS_PATH ="./labeled_v7.loss"
PREDICT_RESULT1 = "./labeled_v7.predict1"
PREDICT_RESULT2 = "./labeled_v7.predict2"
if USE_TOY_DATA == True:
    TRAIN_DATA = "./toy.train"
    TEST_DATA = "./toy.test"
    PREDICT_RESULT = "./toy.predict"
    MODEL_PATH = "./toy_model_lambdarank.ckpt"
MOCK_QUERY_DOC_COUNT = 5
