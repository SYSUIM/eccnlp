# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def do_train(device,
             seed,
             model_dir,
             UIE_model,
             UIE_batch_size,
             max_seq_len,
             init_from_ckpt,
             UIE_learning_rate,
             UIE_num_epochs,
             logging_steps,
             valid_steps,
             save_dir,
             train_data,
             dev_data):
    import argparse
    import time
    import os
    from functools import partial

    import paddle
    from paddlenlp.datasets import MapDataset
    from paddle.utils.download import get_path_from_url
    from paddlenlp.datasets import load_dataset
    from paddlenlp.transformers import AutoTokenizer
    from paddlenlp.metrics import SpanEvaluator
    from paddlenlp.utils.log import logger

    from info_extraction.model import UIE
    from info_extraction.evaluate import evaluate
    from info_extraction.util import set_seed, convert_example, reader, MODEL_MAP

    from utils import get_logger, get_log_path

    # ext_logger = get_logger('ext_logger', get_log_path() + '/ext.log')

    paddle.set_device(device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(seed)

    resource_file_urls = MODEL_MAP[UIE_model]['resource_file_urls']

    # ext_logger.info("Downloading resource files...")
    for key, val in resource_file_urls.items():
        file_path = os.path.join(model_dir, UIE_model, key)
        if not os.path.exists(file_path):
            path = get_path_from_url(val, UIE_model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir + UIE_model)
    model = UIE.from_pretrained(model_dir + UIE_model)

    train_ds = MapDataset(train_data)
    dev_ds = MapDataset(dev_data)

    train_ds = train_ds.map(
        partial(convert_example,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len))
    dev_ds = dev_ds.map(
        partial(convert_example,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len))

    train_batch_sampler = paddle.io.BatchSampler(dataset=train_ds,
                                                 batch_size=UIE_batch_size,
                                                 shuffle=True)
    train_data_loader = paddle.io.DataLoader(dataset=train_ds,
                                             batch_sampler=train_batch_sampler,
                                             return_list=True)

    dev_batch_sampler = paddle.io.BatchSampler(dataset=dev_ds,
                                               batch_size=UIE_batch_size,
                                               shuffle=False)
    dev_data_loader = paddle.io.DataLoader(dataset=dev_ds,
                                           batch_sampler=dev_batch_sampler,
                                           return_list=True)

    if init_from_ckpt and os.path.isfile(init_from_ckpt):
        state_dict = paddle.load(init_from_ckpt)
        model.set_dict(state_dict)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    optimizer = paddle.optimizer.AdamW(learning_rate=UIE_learning_rate,
                                       parameters=model.parameters())

    criterion = paddle.nn.BCELoss()
    metric = SpanEvaluator()

    loss_list = []
    global_step = 0
    best_step = 0
    best_f1 = 0
    tic_train = time.time()
    for epoch in range(1, UIE_num_epochs + 1):
        for batch in train_data_loader:
            input_ids, token_type_ids, pos_ids, att_mask, start_ids, end_ids = batch
            start_prob, end_prob = model(input_ids, token_type_ids, pos_ids,
                                         att_mask)
            start_ids = paddle.cast(start_ids, 'float32')
            end_ids = paddle.cast(end_ids, 'float32')
            loss_start = criterion(start_prob, start_ids)
            loss_end = criterion(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_list.append(float(loss))

            global_step += 1
            if global_step % logging_steps == 0 and rank == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                logger.info(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss_avg,
                       logging_steps / time_diff))
                tic_train = time.time()

            if global_step % valid_steps == 0 and rank == 0:
                save_dir = os.path.join(save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                model_to_save.save_pretrained(save_dir)
                # ext_logger.disable()
                tokenizer.save_pretrained(save_dir)
                # ext_logger.enable()

                precision, recall, f1 = evaluate(model, metric, dev_data_loader)
                logger.info(
                    "Evaluation precision: %.5f, recall: %.5f, F1: %.5f" %
                    (precision, recall, f1))
                if f1 > best_f1:
                    logger.info(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    best_f1 = f1
                    save_dir = os.path.join(save_dir, "model_best")
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                tic_train = time.time()


if __name__ == "__main__":
    import sys
    import logging
    sys.path.append('/data2/panziyang/project/eccnlp/eccnlp')

    device = 'gpu'
    seed = 42
    model_dir = '/data/pzy2022/paddlepaddle/taskflow/'
    UIE_model = 'uie-base'
    UIE_batch_size = 32
    max_seq_len = 512
    init_from_ckpt = None
    UIE_learning_rate = 1e-06
    UIE_num_epochs = 50
    logging_steps = 100
    valid_steps = 100000
    save_dir = '/data/fkj2023/Project/eccnlp/checkpoint/20230325'
    
    from utils import read_list_file
    from data_process.info_extraction import dataset_generate_train

    data = read_list_file('/data/zyx2022/FinanceText/process_file/2.2_raw_dataset_dict_nocut_uni_no.txt')
    train_data, dev_data, test_data = dataset_generate_train(0.8, 0.1, data)
    print(f'train_data: {len(train_data)}, dev_data: {len(dev_data)}, test_data: {len(test_data)}')

    do_train(device = device,
             seed = seed,
             model_dir = model_dir,
             UIE_model = UIE_model,
             UIE_batch_size = UIE_batch_size,
             max_seq_len = max_seq_len,
             init_from_ckpt = init_from_ckpt,
             UIE_learning_rate = UIE_learning_rate,
             UIE_num_epochs = UIE_num_epochs,
             logging_steps = logging_steps,
             valid_steps = valid_steps,
             save_dir = save_dir,
             train_data = train_data,
             dev_data = dev_data)