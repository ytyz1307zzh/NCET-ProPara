'''
 @Date  : 12/11/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import time
print('[INFO] Starting import...')
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
parser.add_argument('-embed_size', type=int, default=128, help="embedding size (including the verb indicator)")
parser.add_argument('-hidden_size', type=int, default=128, help="hidden size of lstm")
parser.add_argument('-epoch', type=int, default=100, help="number of epochs, use -1 to rely on early stopping only")
parser.add_argument('-impatience', type=int, default=20, help='number of evaluation rounds for early stopping')
parser.add_argument('-lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('-dropout', type=float, default=0.1, help="droppout rate")
parser.add_argument('-mode', type=str, default='train', help="train or test")
parser.add_argument('-ckpt_dir', type=str, required=True, help="checkpoint directory")
parser.add_argument('-restore', type=str, default='', help="restoring model path")
parser.add_argument('-report', type=int, default=2, help="report frequence per epoch")
parser.add_argument('-elmo_dir', type=str, default='elmo', help="directory that contains options and weight files for allennlp Elmo")
parser.add_argument('-data_dir', type=str, default='data', help="directory to the train/dev/test data")
parser.add_argument('-debug', action='store_true', default=False, help="enable debug mode, change data files to debug data")
parser.add_argument('-no_cuda', action='store_true', default=False, help="if true, will only use cpu")
opt = parser.parse_args()


# TODO: Implement this save_model function
def save_model(path: str, model):
    pass


def train():

    train_set = ProparaDataset('./data/train.json')
    if opt.debug:
        print('*'*20 + '[INFO] Debug mode enabled. Switch training set to debug.json' + '*'*20)
        train_set = ProparaDataset('./data/debug.json')

    train_batch = DataLoader(dataset = train_set, batch_size = opt.batch_size, shuffle = True, collate_fn = Collate())
    dev_set = ProparaDataset('./data/dev.json')

    if opt.debug:
        print('*'*20 + '[INFO] Debug mode enabled. Switch dev set to debug.json' + '*'*20)
        dev_set = ProparaDataset('./data/debug.json')

    model = NCETModel(batch_size = opt.batch_size, embed_size = opt.embed_size, hidden_size = opt.hidden_size,
                        dropout = opt.dropout, elmo_dir = opt.elmo_dir)
    if not opt.no_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    best_score = np.NINF
    impatience = 0
    epoch_i = 0

    if opt.epoch == -1:
        opt.epoch = np.inf

    while epoch_i < opt.epoch:

        model.train()
        train_instances = len(train_set)
        report_loss, report_accuracy, start_time = [], [], time.time()
        batch_cnt = 0
        if train_instances % opt.batch_size == 0:
            total_batches = train_instances // opt.batch_size
        else:
            total_batches = train_instances // opt.batch_size + 1
        report_freq = train_instances // opt.report  # frequency of reporting results (in batches)

        for batch in train_batch:
            # with open('logs/debug.log', 'w', encoding='utf-8') as debug_file:
            #     torch.set_printoptions(threshold=np.inf)
            #     print(batch, file = debug_file)
            model.zero_grad()

            paragraphs = batch['paragraph']
            char_paragraph = batch_to_ids(paragraphs)
            entity_mask = batch['entity_mask']
            verb_mask = batch['verb_mask']
            loc_mask = batch['loc_mask']
            gold_loc_seq = batch['gold_loc_seq']
            gold_state_seq = batch['gold_state_seq']

            if not opt.no_cuda:
                char_paragraph.cuda()
                entity_mask.cuda()
                verb_mask.cuda()
                loc_mask.cuda()

            train_loss, train_accuracy = model(char_paragraph = char_paragraph, entity_mask = entity_mask, verb_mask = verb_mask,
                                               loc_mask = loc_mask, gold_loc_seq = gold_loc_seq, gold_state_seq = gold_state_seq)

            train_loss.backward()
            optimizer.step()
            report_loss.append(train_loss.item())
            report_accuracy.append(train_accuracy)
            batch_cnt += 1

            # time to report results
            if batch_cnt % report_freq == 0 or batch_cnt == total_batches:

                print(f'{batch_cnt}/{total_batches}, Epoch {epoch_i+1}: training loss: {mean(report_loss):.3f}, '
                      f'training state prediction accuracy: {mean(report_accuracy)*100:.3f}%, time elapse: {time.time()-start_time:.2f}')

                model.eval()
                eval_score = eval(dev_set, model)
                model.train()

                if eval_score > best_score:  # new best score
                    best_score = eval_score
                    impatience = 0
                    print('New best score!')
                    save_model(os.path.join(opt.ckpt_dir, f'best_checkpoint_{best_score:.3f}.pt'), model)
                else:
                    impatience += 1
                    print(f'Impatience: {impatience}, best score: {best_score:.3f}.')
                    save_model(os.path.join(opt.ckpt_dir, f'checkpoint_{eval_score:.3f}.pt'), model)
                    if impatience >= opt.impatience:
                        print('Early Stopping!')
                        quit()

                report_loss, report_accuracy, start_time = [], [], time.time()

        epoch_i += 1



        # summary(model, char_paragraph, entity_mask, verb_mask, loc_mask)
        # with SummaryWriter() as writer:
        #     writer.add_graph(model, (char_paragraph, entity_mask, verb_mask, loc_mask))


def eval(dev_set, model):
    start_time = time.time()
    dev_batch = DataLoader(dataset = dev_set, batch_size = opt.batch_size, shuffle = False, collate_fn = Collate())

    report_loss, report_accuracy = [], []
    with torch.no_grad():
        for batch in dev_batch:
            paragraphs = batch['paragraph']
            char_paragraph = batch_to_ids(paragraphs)
            entity_mask = batch['entity_mask']
            verb_mask = batch['verb_mask']
            loc_mask = batch['loc_mask']
            gold_loc_seq = batch['gold_loc_seq']
            gold_state_seq = batch['gold_state_seq']

            if not opt.no_cuda:
                char_paragraph.cuda()
                entity_mask.cuda()
                verb_mask.cuda()
                loc_mask.cuda()

            eval_loss, eval_accuracy = model(char_paragraph=char_paragraph, entity_mask=entity_mask, verb_mask=verb_mask,
                                            loc_mask=loc_mask, gold_loc_seq=gold_loc_seq, gold_state_seq=gold_state_seq)
            report_loss.append(eval_loss.item())
            report_accuracy.append(eval_accuracy)

    print(f'Evaluation: eval loss: {mean(report_loss):.3f}, '
          f'eval state prediction accuracy: {mean(report_accuracy)*100:.3f}%, time elapse: {time.time()-start_time:.2f}')

    return eval_accuracy



if __name__ == "__main__":

    # train_set = ProparaDataset(os.path.join(opt.data_dir, 'train.json'))
    # dev_set = ProparaDataset(os.path.join(opt.data_dir, 'dev.json'))
    # test_set = ProparaDataset(os.path.join(opt.data_dir, 'test.json'))
    train()

