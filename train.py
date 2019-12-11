'''
 @Date  : 12/11/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import time
import_start_time = time.time()
import torch
import json
import os
import numpy as np
from typing import List, Dict
from Constants import *
import argparse
from torchsummaryX import summary
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from allennlp.modules.elmo import batch_to_ids
from utils import *
from Dataset import *
from Model import *
print(f'[INFO] Import modules time: {time.time() - import_start_time}')


parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-embed_size', type=int, default=256, help="embedding size (including the verb indicator)")
parser.add_argument('-hidden_size', type=int, default=128, help="hidden size of lstm")
parser.add_argument('-epoch', type=int, default=100, help="number of epochs")
parser.add_argument('-impatience', type=int, default=20, help='number of evaluation rounds for early stopping')
parser.add_argument('-lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('-dropout', type=float, default=0.1, help="droppout rate")
parser.add_argument('-mode', type=str, default='train', help="train or test")
parser.add_argument('-ckpt_dir', type=str, default='ckpt', help="checkpoint directory")
parser.add_argument('-restore', type=str, default='', help="restoring model path")
parser.add_argument('-report', type=int, default=2, help="report frequence per epoch")
parser.add_argument('-elmo_dir', type=str, default='elmo', help="directory that contains options and weight files for allennlp Elmo")
parser.add_argument('-data_dir', type=str, default='data', help="directory to the train/dev/test data")
parser.add_argument('-debug', action='store_true', default=False, help="enable debug mode, change data files to debug data")
opt = parser.parse_args()


def train():

    debug_set = ProparaDataset('./data/debug.json')
    debug_batch = DataLoader(dataset = debug_set, batch_size = opt.batch_size, shuffle = True, collate_fn = Collate())

    model = NCETModel(batch_size = opt.batch_size, embed_size = opt.embed_size, hidden_size = opt.hidden_size,
                        dropout = opt.dropout, elmo_dir = opt.elmo_dir)
    model.cuda()

    for batch in debug_batch:
        model.train()

        paragraphs = batch['paragraph']
        char_paragraph = batch_to_ids(paragraphs).cuda()
        print('char_paragraph: ', char_paragraph.size())
        entity_mask = batch['entity_mask'].cuda()
        verb_mask = batch['verb_mask'].cuda()
        loc_mask = batch['loc_mask'].cuda()

        # summary(model, char_paragraph, entity_mask, verb_mask, loc_mask)
        # with SummaryWriter() as writer:
        #     writer.add_graph(model, (char_paragraph, entity_mask, verb_mask, loc_mask))


if __name__ == "__main__":

    # train_set = ProparaDataset(os.path.join(opt.data_dir, 'train.json'))
    # dev_set = ProparaDataset(os.path.join(opt.data_dir, 'dev.json'))
    # test_set = ProparaDataset(os.path.join(opt.data_dir, 'test.json'))
    train()

