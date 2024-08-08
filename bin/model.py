#!/usr/bin/python
#-*- coding:utf8 -*-
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

class ComplexModel(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, num_layers, dropout_rate):
        super(ComplexModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bilstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 4, 256)
        self.fc2 = nn.Linear(256, output_size)            
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, aa_seq, peptide_seq):
        aa_embedded = self.embedding(aa_seq).permute(0, 2, 1)
        peptide_embedded = self.embedding(peptide_seq).permute(0, 2, 1)
        aa_conv = self.pool(torch.relu(self.conv2(torch.relu(self.conv1(aa_embedded)))))
        peptide_conv = self.pool(torch.relu(self.conv2(torch.relu(self.conv1(peptide_embedded)))))
        aa_conv = aa_conv.permute(0, 2, 1)
        peptide_conv = peptide_conv.permute(0, 2, 1)
        _, (aa_h_n, _) = self.bilstm(aa_conv)
        aa_output = torch.cat((aa_h_n[-2], aa_h_n[-1]), dim=1)
        _, (peptide_h_n, _) = self.bilstm(peptide_conv)
        peptide_output = torch.cat((peptide_h_n[-2], peptide_h_n[-1]), dim=1)
        combined_output = torch.cat((aa_output, peptide_output), dim=1)
        combined_output = self.dropout(torch.relu(self.fc1(combined_output)))
        output = self.fc2(combined_output)
        return output

