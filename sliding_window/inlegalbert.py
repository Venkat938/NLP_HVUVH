# -*- coding: utf-8 -*-
"""InLegalBERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RBSeYMtmge_I5EFxBSYsTsgLmWj8Nqer
"""

import pandas as pd

train_data = pd.read_csv("train_data.csv")

test_data = pd.read_csv("test_data.csv")





# !pip install simpletransformers

from simpletransformers.classification import ClassificationModel, ClassificationArgs

# !pip install huggingface_hub

from huggingface_hub import notebook_login

notebook_login()

import torch

model_args = ClassificationArgs(sliding_window = True,stride = 0.8)
model_args.train_batch_size = 16
model_args.eval_batch_size = 16


cuda_available = torch.cuda.is_available()

model = ClassificationModel('bert', 'law-ai/InLegalBERT', num_labels=2,weight=[1,1.5],use_cuda = cuda_available,use_auth_token=True , args = model_args)

model.train_model(train_data)

from sklearn.metrics import f1_score, accuracy_score


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')
    
result, model_outputs, wrong_predictions = model.eval_model(test_data, f1=f1_multiclass, acc=accuracy_score)

result