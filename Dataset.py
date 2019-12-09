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
import numpy as np
from typing import List
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


    def get_mask(self, mention_idx: List[int], para_len: int) -> List[int]:
        """
        Given a list of mention positions of the entity/verb/location in a paragraph,
        compute the mask of it.
        """
        return [1 if i in mention_idx else 0 for i in range(para_len)]


    def __getitem__(self, index: int):

        instance = self.dataset[index]

        entity_name = instance['entity']  # used in the evaluation process
        para_id = instance['id']  # used in the evaluation process
        total_tokens = instance['total_tokens']  # used in compute mask vector
        total_sents = instance['total_sents']
        total_loc_cands = instance['total_loc_candidates']
        loc_cand_list = instance['loc_cand_list']

        metadata = {'para_id': para_id,
                    'entity': entity_name,
                    'total_sents': total_sents,
                    'total_loc_cands': total_loc_cands
                    }
        paragraph = instance['paragraph'].strip().split()  # Elmo processes list of words     
        gold_state_seq = torch.IntTensor([self.state2idx[label] for label in instance['gold_state_seq']])

        loc2idx = {loc_cand_list[idx]: idx for idx in range(total_loc_cands)}
        loc2idx['-'] = NIL_LOC
        loc2idx['?'] = UNK_LOC
        # note that the loc_cand_list in exactly "idx2loc" (excluding '?' and '-')
        gold_loc_seq = torch.IntTensor([loc2idx[loc] for loc in instance['gold_loc_seq']])

        sentence_list = instance['sentence_list']
        # (num_sent, num_tokens)
        entity_mask_list = torch.IntTensor([self.get_mask(sent['entity_mention'], total_tokens) for sent in sentence_list])
        # (num_sent, num_tokens)
        verb_mask_list = torch.IntTensor([self.get_mask(sent['verb_mention'], total_tokens) for sent in sentence_list])
        # (num_cand, num_sent, num_tokens)
        loc_mask_list = torch.IntTensor([[self.get_mask(sent['loc_mention_list'][idx], total_tokens) for sent in sentence_list]
                                            for idx in range(total_loc_cands)])

        sample = {'metadata': metadata,
                  'paragraph': paragraph,
                  'gold_loc_seq': gold_loc_seq,
                  'gold_state_seq': gold_state_seq,
                  'entity_mask': entity_mask_list,
                  'verb_mask': verb_mask_list,
                  'loc_mask': loc_mask_list
                }

        return sample

        
# debug script
# torch.set_printoptions(threshold=np.inf)
dev_dataset = ProparaDataset('data/dev.json', is_train = False)
instance = dev_dataset[0]
print(instance)
