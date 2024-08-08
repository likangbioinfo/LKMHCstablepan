#!/usr/bin/python
#-*- coding:utf8 -*-
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model import ComplexModel
import argparse
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

input_size = 20
embedding_dim = 64
hidden_size = 256
output_size = 1
num_layers = 3
dropout_rate = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ComplexModel(input_size, embedding_dim, hidden_size, output_size, num_layers, dropout_rate).to(device)

model.load_state_dict(torch.load('{}/../model/best_model.pth'.format(script_dir)))
model.eval()


def predict(model, aa_seq, peptide_seq, device):
    model.eval()
    with torch.no_grad():
        aa_seq_tensor = torch.tensor(aa_seq, dtype=torch.long).unsqueeze(0).to(device)
        peptide_seq_tensor = torch.tensor(peptide_seq, dtype=torch.long).unsqueeze(0).to(device)
        output = model(aa_seq_tensor, peptide_seq_tensor)
        return output.item()

def pseudosequences_dic(pseudosequences_csv):
    df=pd.read_csv(pseudosequences_csv)
    dic=dict(zip(df['allele'], df['pseudosequence']))
    return dic



def get_stable_score(input_file,output_file,pseudosequences_dict):
    amino_acid_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    with open(input_file) as ff,open(output_file,"w") as f:
        for line in ff:
            tmps=line.strip().split("\t")
            if tmps[0]=="HLA":
                f.write(line.strip()+"\t"+"Stable_score\n")
            elif tmps[0] not in pseudosequences_dict:
                print("{0}, the format of HLA is wrong!".format(tmps[0]))
                exit(1)
            else:
                pseudosequences=[amino_acid_dict[aa] for aa in pseudosequences_dict[tmps[0]]]
                pep=[amino_acid_dict[aa] for aa in tmps[1]]
                max_len=max(len(pseudosequences),len(pep))
                pseudosequences=np.pad(pseudosequences, (0, max_len - len(pseudosequences)), 'constant')
                pep=np.pad(pep, (0, max_len - len(pep)), 'constant')
                predicted_stable_score = predict(model, pseudosequences, pep, device)
                f.write(line.strip()+"\t"+str(predicted_stable_score)+"\n")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i',help='the table contained HLA and peptides',dest='input',required=True)
    parser.add_argument('-o',help='the output table',dest='output',required=True)
    args=parser.parse_args()
    pseudosequences_dict=pseudosequences_dic("{}/../data/class1_pseudosequences_new.csv".format(script_dir))
    get_stable_score(args.input,args.output,pseudosequences_dict)


if __name__=="__main__":
    main()
