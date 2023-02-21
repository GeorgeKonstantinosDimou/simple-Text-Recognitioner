import torch
import torch.nn as nn


class Recognizer(nn.Module):

    def __init__(self, params, flag):
        super(Recognizer, self).__init__()
        self.num_layers = params['num_layers']
        self.feature_layers = params['feature_layers']
        self.hidden_dim = params['hidden_dim']
        self.vocab_size = params['vocab_size']
        #self.START_TOKEN = params['vocab_size'] + 1
        self.END_TOKEN = params['vocab_size'] + 1
        self.out_seq_len = params['out_seq_len']
        self.device = params['device']
        self.batch_size = params['batch_size']

        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.feature_layers, self.hidden_dim, self.num_layers)
        self.drop = nn.Dropout(p = 0.2)
        self.linear_out = nn.Linear(self.hidden_dim,
                                    self.vocab_size + 2)  # this is essentially the linear transformation for the decoder
        
        self.flag = flag

    def forward(self, conv_f, target=None):
        self.batch_size = conv_f.shape[0]
        # if self.flag:
        #     self.maxpool1 = nn.MaxPool2d((conv_f.size(dim = 2), 1), stride=(conv_f.size(dim = 2), 1))
        self.h0 = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))
        self.c0 = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))
        
        x = conv_f #[batch_size, feature_maps, seq_length]
        if self.flag:
            #x = conv_f.view(self.batch_size, self.feature_layers, -1)
            x = torch.squeeze(conv_f)
            x = x.permute(2, 0, 1)  #[heght/width after maxpool, -1, features]
            x = self.drop(x)
        else:
            x = x.permute(2, 0, 1) #[seq_length, batch_size, features_maps]

        output, (self.h0, self.c0) = self.lstm(x) # [seq_length, batch_size, hidden_size]
        output = self.relu(output)
        output = self.linear_out(output) #[seq_length, batch_size, vocab_size + 2]
        
        return output