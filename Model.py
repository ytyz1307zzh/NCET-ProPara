'''
 @Date  : 12/11/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import torch
import torch.nn as nn
import json
import os
import time
import numpy as np
from typing import List, Dict
from Constants import *
from utils import *
from allennlp.modules.elmo import Elmo, batch_to_ids


class NCETModel(nn.Module):

    def __init__(self, batch_size: int, embed_size: int, hidden_size: int, dropout: float, elmo_dir: str):

        super(NCETModel, self).__init__()
        self.EmbeddingLayer = NCETEmbedding(batch_size = batch_size, embed_size = embed_size,
                                            elmo_dir = elmo_dir, dropout = dropout)
        self.TokenEncoder = nn.LSTM(input_size = embed_size, hidden_size = hidden_size,
                                    num_layers = 1, batch_first = True, dropout = dropout, bidirectional = True)
        

    def forward(self, paragraphs: List, entity_mask: torch.IntTensor, 
                verb_mask: torch.IntTensor, loc_mask: torch.IntTensor):

        embeddings = self.EmbeddingLayer(paragraphs, verb_mask)  # (batch, max_tokens, embed_size)
        token_rep, _ = self.TokenEncoder(embeddings)
        

    
class NCETEmbedding(nn.Module):

    def __init__(self, batch_size: int, embed_size: int, elmo_dir: str, dropout: float):

        super(NCETEmbedding, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.options_file = os.path.join(elmo_dir, 'elmo_2x4096_512_2048cnn_2xhighway_options.json')
        self.weight_file = os.path.join(elmo_dir, 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
        self.elmo = Elmo(self.options_file, self.weight_file, num_output_representations=1, requires_grad=False,
                            do_layer_norm=False, dropout=0)
        self.embed_project = Linear(1024, self.embed_size - 1, dropout = dropout)  # 1024 is the default size of Elmo, leave 1 dim for verb indicator


    def forward(self, paragraphs: List, verb_mask: torch.IntTensor):
        """
        Args: 
            paragraphs - unpadded paragraphs, size (batch, tokens(unidentical))
            verb_mask - size (batch, max_sents, max_tokens)
        Return:
            embeddings - token embeddings, size (batch, max_tokens, embed_size)
        """
        max_tokens = max([len(para) for para in paragraphs])
        elmo_embeddings = self.get_elmo(paragraphs, max_tokens)
        elmo_embeddings = self.embed_project(elmo_embeddings)
        verb_indicator = self.get_verb_indicator(verb_mask, max_tokens)
        embeddings = torch.cat([elmo_embeddings, verb_indicator], dim = -1)

        assert embeddings.size() == (self.batch_size, max_tokens, self.embed_size)
        return embeddings


    def get_elmo(self, paragraphs: List, max_tokens: int):
        """
        Compute the Elmo embedding of the paragraphs.
        Return:
            Elmo embeddings, size(batch, max_tokens, elmo_embed_size=1024)
        """
        character_ids = batch_to_ids(paragraphs).cuda()
        # embeddings['elmo_representations'] is a list of tensors with length 'num_output_representations' (here it = 1)
        elmo_embeddings = self.elmo(character_ids)['elmo_representations'][0]  # (batch, max_tokens, elmo_embed_size=1024)
        assert elmo_embeddings.size() == (self.batch_size, max_tokens, 1024)
        return elmo_embeddings

    
    def get_verb_indicator(self, verb_mask, max_tokens: int):
        """
        Get the binary scalar indicator for each token
        """
        verb_indicator = torch.sum(verb_mask, dim = 1, dtype = torch.float).unsqueeze_(dim = -1)
        assert verb_indicator.size() == (self.batch_size, max_tokens, 1)
        return verb_indicator


class Linear(nn.Module):
    ''' 
    Simple Linear layer with xavier init 
    '''
    def __init__(self, d_in: int, d_out: int, dropout: float, bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(p = dropout)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.dropout(self.linear(x))