'''
 @Date  : 12/11/2019
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
import argparse
from torch.utils.data import DataLoader
from utils import *
from Dataset import *


debug_set = ProparaDataset('./data/debug.json')
debug_batch = DataLoader(dataset = debug_set, batch_size = 2, shuffle = True, collate_fn = Collate())

with open('logs/debug.log', 'w', encoding='utf-8') as debug_file:
    for batch in debug_batch:
        print(batch, file = debug_file)
