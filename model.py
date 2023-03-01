import torch
import torch.nn as nn
import torch.functional as F
from transformers import BertPreTrainedModel, BertModel
class BERT_BiLSTM(BertPreTrainedModel):

    def __init__(self, config, need_birnn=True,num_rel =5, rnn_dim=128, ):
        super(BERT_BiLSTM, self).__init__(config)
        self.loss_function = nn.CrossEntropyLoss() 
        self.num_rel = num_rel #how many tags (B-xx, I-xx, O) are used for NER
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_birnn = need_birnn

        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim*2 # x2 due to bidirectional
        
        self.layer = nn.Linear(out_dim, self.num_rel)

    

    def forward(self, input_ids,  token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        #output[1] has shape [batch,768], sentence embedding
        #output[0] has shape [batch,512,768], 512 is the length of padded sentence
        output_ = outputs[0] #batch,512,768
        if self.need_birnn:
            output_, _ = self.birnn(output_) #batch, 512, 256
        output_ = self.dropout(output_)
        output_= self.layer(output_) #batch, num_rel
        return output_

    
    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        output_ = outputs[0]
        if self.need_birnn:
            output_, _ = self.birnn(output_)
        output_ = self.dropout(output_) # call model.eval() before this predict will ignore the dropout
        output_= self.layer(output_)

        return output_
