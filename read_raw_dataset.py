'''
 @Date  : 12/02/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

"""
Reads the raw csv files downloaded from the Propara website, then create JSON files which contain
lists of instances in train, dev and test

JSON format: a list of instances
Instance
    |____topic
    |____prompt
    |____paragraph id
    |____paragraph (string)
    |____entity (string)
    |____gold state change sequence (list of labels)
    |____number of sentences in the paragraph
    |____number of location candidates
    |____gold locations (list of strings, len = sent + 1)
    |____list of sentences
               |____sentence id
               |____entity mask (all 0 if not exist)
               |____verb mask (all 0 if not exist)
               |____list of location candidates
                            |____location mask (all 0 if not exist) 
                                (list length equal to number of location candidates)
"""

from typing import Dict, List
import pandas as pd
import argparse
import json
pd.set_option('display.max_columns', 50)


def read_paragraph(filename: str) -> Dict[int, Dict]:

    csv_data = pd.read_csv(filename)
    paragraph_result = {}
    max_sent = len(csv_data.columns) - 3  # should equal to 10 in this case

    for _, row in csv_data.iterrows():
        para_id = int(row['Paragraph ID'])
        topic = row['Topic']
        prompt = row['Prompt']
        sent_list = []

        for i in range(1, max_sent + 1):
            sent = row[f'Sentence{i}']
            if pd.isna(sent):
                break
            sent_list.append(sent)

        text = ' '.join(sent_list)
        paragraph_result[para_id] = {'id': para_id,
                                     'topic': topic,
                                     'prompt': prompt,
                                     'paragraph': text}
    
    print(f'Paragraphs read: {len(paragraph_result)}')
    return paragraph_result


def read_annotation(filename: str, paragraph_result: Dict[int, Dict], train: bool) -> Dict[int, Dict]:
    """
    1. read csv
    2. get the entities
    3. extract location candidates
    4. for each entity, create an instance for it
    5. read the entity's initial state
    5. read each sentence, give it an ID
    6. compute entity mask (length of mask vector = length of paragraph)
    7. extract the nearest verb to the entity, compute verb mask
    8. for each location candidate, compute location mask
    9. read entity's state at current timestep
    10. for the training set, if gold location is not extracted in step 3, add it to the candidate set. Back to step 6
    11. reading ends, compute the number of sentences
    12. get the number of location candidates
    13. infer the gold state change sequence
    """


def read_split(filename: str, paragraph_result: Dict[int, Dict]):

    train_para, dev_para, test_para = {}, {}, {}
    csv_data = pd.read_csv(filename)

    for _, row in csv_data.iterrows():

        para_id = int(row['Paragraph ID'])
        para_data = paragraph_result[para_id]
        partition = row['Partition']
        if partition == 'train':
            train_para[para_id] = para_data
        elif partition == 'dev':
            dev_para[para_id] = para_data
        elif partition == 'test':
            test_para[para_id] = para_data
        
    print('Number of train paragraphs: ', len(train_para))
    print('Number of dev paragraphs: ', len(dev_para))
    print('Number of test paragraphs: ', len(test_para))

    return train_para, dev_para, test_para


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-para_file', type=str, default='../data/Paragraphs.csv', help='path to the paragraph csv')
    parser.add_argument('-state_file', type=str, default='../data/State_change_annotations.csv', 
                        help='path to the state annotation csv')
    parser.add_argument('-split_file', type=str, default='../data/train_dev_test.csv', 
                        help='path to the csv that annotates the train/dev/test split')
    opt = parser.parse_args()

    print('Received arguments:')
    print(opt)
    print('-' * 50)

    paragraph_result = read_paragraph(opt.para_file)
    train_para, dev_para, test_para = read_split(opt.split_file, paragraph_result)
    train_instances = read_annotation(opt.state_file, train_para, train = True)
    dev_instances = read_annotation(opt.state_file, dev_para, train = False)
    test_instances = read_annotation(opt.state_file, test_para, train = False)

