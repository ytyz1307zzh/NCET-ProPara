'''
 @Date  : 12/02
 9/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import torch
import json
import os
import time
from Constants import *


class ProparaDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: str, is_train: bool):
        super(ProparaDataset, self).__init__()

        print('Starting load...')
        print(f'Load data from {data_path}')
        start_time = time.time()

        self.dataset = json.load(open(data_path, 'r', encoding='utf-8'))
        self.state2idx = state2idx
        self.idx2state = idx2state

        print(f'{len(self.dataset)} instances of data loaded. Time Elapse: {time.time() - start_time}')
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        instance = self.dataset[index]
        sample = {}

        entity_name = instance['entity']  # used in the evaluation process
        para_id = instance['id']  # used in the evaluation process
        total_tokens = instance['total_tokens']  # used in compute mask vector
        total_sents = instance['total_sents']
        total_loc_cands = instance['total_loc_candidates']
        loc_cand_list = instance['loc_cand_list']
        sentence_list = instance['sentence_list']

        metadata = {'para_id': para_id,
                    'entity': entity_name}
        paragraph = instance['paragraph'].strip().split()  # Elmo processes list of words     
        gold_state_seq = torch.IntTensor([self.state2idx[label] for label in instance['gold_state_seq']])

        loc2idx = {loc_cand_list[idx]: idx for idx in range(total_loc_cands)}
        loc2idx['-'] = NIL_LOC
        loc2idx['?'] = UNK_LOC
        # note that the loc_cand_list in exactly "idx2loc"
        gold_loc_seq = torch.IntTensor([loc2idx[loc] for loc in instance['gold_loc_seq']])



        
# debug script
dev_dataset = ProparaDataset('data/dev.json', is_train = False)
instance = dev_dataset[0]
