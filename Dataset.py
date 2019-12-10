'''
 @Date  : 12/09/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import torch
import json
import os
import time
import numpy as np
from typing import List, Dict
from Constants import *


class ProparaDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: str):
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

        print('total_loc_cands: ', total_loc_cands)

        metadata = {'para_id': para_id,
                    'entity': entity_name,
                    'total_sents': total_sents,
                    'total_loc_cands': total_loc_cands
                    }
        paragraph = instance['paragraph'].strip().split()  # Elmo processes list of words     
        assert len(paragraph) == total_tokens
        gold_state_seq = torch.IntTensor([self.state2idx[label] for label in instance['gold_state_seq']])

        loc2idx = {loc_cand_list[idx]: idx for idx in range(total_loc_cands)}
        loc2idx['-'] = NIL_LOC
        loc2idx['?'] = UNK_LOC
        # note that the loc_cand_list in exactly "idx2loc" (excluding '?' and '-')
        gold_loc_seq = torch.IntTensor([loc2idx[loc] for loc in instance['gold_loc_seq']])[1:]  # won't predict initial location (step 0)

        assert gold_loc_seq.size() == gold_state_seq.size()
        sentence_list = instance['sentence_list']
        assert total_sents == len(sentence_list)

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


class Collate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def collate(self, batch: List[Dict]) -> List[Dict]:
        """
        args:
            batch - list of instances constructed by dataset

        reutrn:
            batch - list of padded instances
        """
        # find max number of sentences & tokens
        max_sents = max([inst['metadata']['total_sents'] for inst in batch])
        max_tokens = max([len(inst['paragraph']) for inst in batch])
        print('max_sents: ', max_sents)
        print('max_tokens: ', max_tokens)

        # pad according to max_len
        batch = list(map(lambda x: self.pad_instance(x, max_sents = max_sents, max_tokens = max_tokens), batch))
        return batch

    
    def pad_instance(self, instance: Dict, max_sents: int, max_tokens: int) -> Dict:
        """
        args: instance - instance to pad
        max_sents: maximum number of sentences in this batch
        max_tokens: maximum number of tokens in this batch
        """
        instance['gold_state_seq'] = self.pad_tensor(instance['gold_state_seq'], pad = max_sents, dim = 0)
        instance['gold_loc_seq'] = self.pad_tensor(instance['gold_loc_seq'], pad = max_sents, dim = 0, pad_val = PAD_LOC)
        
        instance['entity_mask'] = self.pad_mask_list(instance['entity_mask'], max_sents = max_sents, max_tokens = max_tokens)
        instance['verb_mask'] = self.pad_mask_list(instance['verb_mask'], max_sents = max_sents, max_tokens = max_tokens)
        instance['loc_mask'] = torch.stack(list(map(lambda x: self.pad_mask_list(x, max_sents = max_sents, max_tokens = max_tokens), 
                                        instance['loc_mask'])))

        return instance

    
    def pad_mask_list(self, vec: torch.Tensor, max_sents: int, max_tokens: int) -> torch.Tensor:
        """
        Pad a tensor of mask list
        """
        tmp_vec = self.pad_tensor(vec, pad = max_tokens, dim = 1)
        return self.pad_tensor(tmp_vec, pad = max_sents, dim = 0)

    
    def pad_tensor(self, vec: torch.Tensor, pad: int, dim: int, pad_val: int = 0) -> torch.Tensor:
        """
        args:
            vec - tensor to pad
            pad - the size to pad to
            dim - dimension to pad

        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        pad_size = list(vec.size())
        pad_size[dim] = pad - vec.size(dim)
        pad_vec = torch.zeros(*pad_size, dtype=vec.dtype)

        if pad_val != 0:
            pad_vec.fill_(pad_val)

        return torch.cat([vec, pad_vec], dim=dim)


    def __call__(self, batch):
        return self.collate(batch)

        
