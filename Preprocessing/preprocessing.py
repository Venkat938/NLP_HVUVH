#!/usr/bin/env python
# coding: utf-8

import pandas as pd

dataframe = pd.read_csv("C:/Users/venkat/Downloads/ILDC/Data/ILDC_single/ILDC_single/ILDC_single.csv/ILDC_single.csv")

dataframe_pos = dataframe[dataframe['label'] == 0]

dataframe.head()

train_dataset = dataframe[dataframe['split'] == 'train']

test_dataset = dataframe[dataframe['split'] == 'test']

val_dataset = dataframe[dataframe['split'] == 'dev']




test_dataset.shape


train_dataset.drop(['split', 'name'], axis = 1, inplace = True)
test_dataset.drop(['split', 'name'], axis = 1, inplace = True)
val_dataset.drop(['split', 'name'], axis = 1, inplace = True)


train_dataset.to_csv("train_text.csv")
test_dataset.to_csv("test_text.csv")
val_dataset.to_csv("val_text.csv")
