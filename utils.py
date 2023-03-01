import pandas as pd
import torch
from torch.utils.data import TensorDataset
import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np
class EvalObject(object):
    def __init__(self):
        self.accuracy = 0    
        self.precision = 0     
        self.recall= 0    
        self.f1 = 0

class InputFeatures(object):
    # """A single set of features of data."""
    def __init__(self, input_ids, segment_ids, input_mask, label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


def get_label_map(filepath):
    with open(rel2id_path,"r", encoding="utf-8") as f:
        rel2id = json.load(f)
    values = list(rel2id.values())
    
    id2rel = dict()
    rel2id = dict()
    rel2id = {k: v for v, k in enumerate(values)}
    id2rel = {v: k for v, k in enumerate(values)}
    return rel2id , id2rel


def get_data_from_jsonObject(tokenizer, jsonObject, mapper, padding = True, max_length = 512):
    
    label = mapper[jsonObject["relation"]]

    cls_token = [tokenizer.cls_token]
    sep_token = [tokenizer.sep_token]
    tokens = jsonObject["token"]
    e1_marking_start = ["[E1]"]
    e1_marking_end = ["[/E1]"]
    e2_marking_start = ["[E2]"]
    e2_marking_end= ["[/E2]"]
    h_pos = jsonObject["h"]["pos"]
    t_pos = jsonObject["t"]["pos"]
    before_e1 =cls_token + tokens[0:h_pos[0]]
    e1_entity = e1_marking_start+ tokens[h_pos[0]:h_pos[1]] + e1_marking_end
    between_e1_e2 = tokens[h_pos[1]:t_pos[0]]
    e2_entity = e2_marking_start+ tokens[t_pos[0]:t_pos[1]] + e2_marking_end
    after_e2 = tokens[t_pos[1]:] + sep_token
    if h_pos[0] < t_pos[0]:
        before_e1 =cls_token + tokens[0:h_pos[0]]
        e1_entity = e1_marking_start+ tokens[h_pos[0]:h_pos[1]] + e1_marking_end
        between_e1_e2 = tokens[h_pos[1]:t_pos[0]]
        e2_entity = e2_marking_start+ tokens[t_pos[0]:t_pos[1]] + e2_marking_end
        after_e2 = tokens[t_pos[1]:] + sep_token
        tokens = before_e1 + e1_entity + between_e1_e2 + e2_entity + after_e2
    else:
        before_e2 =cls_token + tokens[0:t_pos[0]]
        e1_entity = e1_marking_start+ tokens[h_pos[0]:h_pos[1]] + e1_marking_end
        between_e2_e1 = tokens[t_pos[1]:h_pos[0]]
        e2_entity = e2_marking_start+ tokens[t_pos[0]:t_pos[1]] + e2_marking_end
        after_e1 = tokens[h_pos[1]:] + sep_token
        tokens = before_e2 + e2_entity + between_e2_e1 + e1_entity + after_e1
    
    sequence_ids = [0]*512
    mask_ids = [0]*512
    input_ids = list()
    input_ids.append(101)
    for i in range(0,len(tokens)):
        split = tokenizer(tokens[i],padding= False, max_length = 30, truncation=True)
        temp = [j for j in split["input_ids"] if j not in [101,102]]
        input_ids.extend(temp)
    input_ids.append(102)
    if len(input_ids) < 512:
        input_ids.extend([0]* (512 - len (input_ids)))
    for i in range(0, len(input_ids)):
        if input_ids[i] == 0:
            mask_ids[i] = 0
        else:
            mask_ids[i] = 1
    
    return InputFeatures(input_ids, sequence_ids, mask_ids, label)

def get_data_from_file(tokenizer, file_path, rel2id ):
    
   
    
    with open(file_path,"r",encoding="utf-8") as f:
        texts = f.readlines()
    features = list()
    for s in texts:
        jsonObject = json.loads(s)
        feature = get_data_from_jsonObject(tokenizer,s,mapper=rel2id)
        features.append(feature)
    return features

def get_Dataset(tokenizer, file_path, rel2id ):
    features = get_data_from_file(tokenizer,file_path,rel2id)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids,  all_segment_ids,all_input_mask, all_label_ids)
    return data





from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
# s = """{"token": ["Merpati", "flight", "106", "departed", "Jakarta", "(", "CGK", ")", "on", "a", "domestic", "flight", "to", "Tanjung", "Pandan", "(", "TJQ", ")", "."], "h": {"name": "tjq", "id": "Q1331049", "pos": [16, 17]}, "t": {"name": "tanjung pandan", "id": "Q3056359", "pos": [13, 15]}, "relation": "P931"}"""
# jsonObject = json.loads(s)
# data, sequence_ids, mask_ids, relation = get_data_from_jsonObject(tokenizer= tokenizer, jsonObject= jsonObject)

# decode = tokenizer.convert_ids_to_tokens(data)
# print(decode[0:50])
rel2id_path = r"G:\My Drive\Lab working\Knowledge_from_text\Relation Extraction\input\rel2wiki.json"
file_path = r"G:\My Drive\Lab working\Knowledge_from_text\Relation Extraction\input\wiki80_modified_train.txt"
get_Dataset(tokenizer,file_path,rel2id_path)

