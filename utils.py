'''
 @Date  : 12/09/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import json

max_num_candidates = 0
max_num_tokens = 0
max_num_sents = 0

def count_maximum():

    def load_data(json_file):
        global max_num_candidates
        global max_num_tokens
        global max_num_sents
        data = json.load(open(json_file, 'r', encoding='utf-8'))
        print('number of instances: ', len(data))
        for instance in data:
            num_candidates = instance['total_loc_candidates']
            num_tokens = instance['total_tokens']
            num_sents = instance['total_sents']
            if num_candidates > max_num_candidates:
                max_num_candidates = num_candidates
            if num_tokens > max_num_tokens:
                max_num_tokens = num_tokens
            if num_sents > max_num_sents:
                max_num_sents = num_sents

    load_data('./data/train.json')
    print('train')
    load_data('./data/dev.json')
    print('dev')
    load_data('./data/test.json')
    print('test')

    print(f'max number of candidates: {max_num_candidates}')
    print(f'max number of tokens: {max_num_tokens}')
    print(f'max number of sentences: {max_num_sents}')

