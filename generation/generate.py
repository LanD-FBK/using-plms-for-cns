import os
import json
import csv
import pandas as pd
import time
import random
from collections import Counter
from statistics import mean, median, stdev
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import gc
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from transformers import EncoderDecoderModel, BertTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


    
batch_size=20

def load_model(model_type, model_path):
    # load model
    if model_type == 'GPT-2':
        model = GPT2LMHeadModel.from_pretrained(model_path)
    if model_type == 'DialoGPT':
        model = AutoModelForCausalLM.from_pretrained(model_path)
    if model_type == 'BERT':
        model = EncoderDecoderModel.from_pretrained(model_path)
    if model_type == 'BART':
        model = BartForConditionalGeneration.from_pretrained(model_path)
    if model_type == 'T5':
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    return model
        
def load_tokenizer(model_type, model_path):
    if model_type == 'GPT-2':
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, return_tensors = 'pt')
        # this is necessary in order to pad
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    if model_type == 'DialoGPT':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    if model_type == 'BERT':
        tokenizer = BertTokenizer.from_pretrained(model_path)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    if model_type == 'BART':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', return_tensors='pt')
    if model_type == 'T5':
        tokenizer = AutoTokenizer.from_pretrained('t5-base', return_tensors = 'pt')
    return tokenizer
        

def generate_cn(model_type, model_path, list_hs, decoding, p=0.92, k=40, nb=5, rp=2.0, max_len =156):
    """
    Takes a list of HS in input and outputs a list of lists with 5 generated CNs for each input HS
    """
    model = load_model(model_type, model_path)
    new_cns=[]
    for i in range(len(list_hs)):
        encoded_hs_ids = encode_hs(list_hs[i], model_type, model_path)

        if decoding == 'top-p':
            encoded_new_cn = model.generate(input_ids = encoded_hs_ids,
                                            max_length=max_len,
                                            do_sample=True,
                                            top_p=p,
                                            num_return_sequences = 5)

        if decoding == 'top-k':
            encoded_new_cn = model.generate(input_ids = encoded_hs_ids,
                                            max_length = max_len, 
                                            do_sample = True, 
                                            top_k = k, 
                                            num_return_sequences = 5)

        if decoding == 'beam-search':
            encoded_new_cn = model.generate(input_ids = encoded_hs_ids,
                                            max_length=max_len,
                                            num_beams=nb, 
                                            early_stopping=True, 
                                            num_return_sequences=5, 
                                            repetition_penalty=rp,
                                            do_sample=True)

        if decoding == 'k-p':
            encoded_new_cn = model.generate(input_ids = encoded_hs_ids,
                                            max_length = max_len, 
                                            do_sample = True, 
                                            top_k = k, 
                                            top_p = p,
                                            num_return_sequences = 5)

        new_cns.append(decode_cns(encoded_new_cn, model_type, model_path))

    return new_cns


def encode_hs(input_hs, model_type, model_path, max_len =156):
    """
     takes one HS and returns input IDs of encoded and preprocessed HS (needed for generation)
    """
    tokenizer = load_tokenizer(model_type, model_path)
    if model_type == 'GPT-2':
        processed_hs = f"<hatespeech> {input_hs} <counternarrative>"
        encoded_hs = tokenizer(processed_hs,
                                truncation=True,
                                padding=True,
                                return_tensors="pt").input_ids.to(device)

    if model_type == 'DialoGPT':
        processed_hs = input_hs + tokenizer.eos_token
        encoded_hs = tokenizer.encode(processed_hs, 
                                        return_tensors='pt').to(device)
    
    if model_type in ['BERT', 'BART', 'T5']:
        encoded_hs = tokenizer(input_hs, 
                                return_tensors='pt',
                                max_length = max_len,
                                padding='max_length', 
                                truncation=True).to(device).input_ids.to(device)
    return encoded_hs

def decode_cns(encoded_cns, model_type, model_path):
    """
    takes 5 encoded CNs and returns 5 clean decoded CNs
    """
    tokenizer = load_tokenizer(model_type, model_path)
    clean_gen = []
    if model_type == 'GPT-2':
        decoded_cns = tokenizer.batch_decode(encoded_cns, skip_special_tokens=False)

        for j in range(len(decoded_cns)):
            c = decoded_cns[j].split('<counternarrative>')
            s = c[1].lstrip().strip(' <|endoftext|>')
            if pd.isnull(s):
                clean_gen.append('')
            else:
                clean_gen.append(s)
    
    if model_type == 'DialoGPT':
        clean_gen = [c.split('<|endoftext|>')[1] for c in tokenizer.batch_decode(encoded_cns, skip_special_tokens=False)]

    if model_type in ['BERT', 'BART', 'T5']:
        clean_gen = tokenizer.batch_decode(encoded_cns, skip_special_tokens=True)
    
    return clean_gen


def prepare_output_df(generated_cns, input_data, model_type, decoding):
    """
    takes generated CNs, input data, model type and decoding mechanism 
    and returns a copy of the input data with a new column for each generated CN
    """
    diz_cols = {'beam-search':'bs', 'top-p':'tp', 'top-k':'tk', 'k-p':'kp'}
    output_df = input_data.copy()
    name_col = model_type + '_' + diz_cols[decoding]
    for i in range(5):
        toappend= [g[i] for g in generated_cns]
        output_df[name_col+str(i)] = toappend
    return output_df