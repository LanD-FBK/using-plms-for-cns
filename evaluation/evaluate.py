from datasets import load_dataset, Dataset
from itertools import product
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import transformers
import datasets
from transformers import AutoTokenizer, TFT5ForConditionalGeneration
import os
import pandas as pd
import warnings
import operator
import random
from datasets import load_metric
import string
import spacy
import statistics
from functools import reduce
warnings.filterwarnings('ignore')

rouge = load_metric("rouge", seed=0)
bleu = load_metric("bleu", seed = 0)
en_nlp = spacy.load('en_core_web_sm')

def evaluator(generated_data, metric, reference_column = 'CN_ed', index_column = 'index'): #takes data
  """
  input
  generated_data: input data containing the generated data and the reference data. column names of generated data should be of the type MODEL_decodingnr (i.e. DIALO_kp0, GPT2_tk0,...)
  can take also a dataset with generations from different models
  metric: a string, either 'rouge', 'bleu-1', 'bleu-3' or 'bleu-4'
  reference_column is the name of the column containing the reference text against which to compute the metrics
  index_column is the name of the column containing the index of the CN

  output
  the input dataframe is extended with one column per generation containing the related rouge/bleu score
  """
  generated_data = generated_data.fillna('')
  data = generated_data.copy()

  tf.random.set_seed(0)
  cols = [c for c in generated_data.columns if type(generated_data[c][0])==str and c not in [index_column, reference_column]]
  data = data[cols]
  data[index_column] = generated_data[index_column]

  if metric == 'rouge':
      rouge = load_metric("rouge", seed=0)

      for i in range(len(data)):
          for c in cols:
              rouge.add(prediction = generated_data[c][i], reference = generated_data[reference_column][i])
              rouge_output = rouge.compute(rouge_types=['rougeL'], use_aggregator=False)
              data[c][i] = rouge_output['rougeL'][0][2] #fmeasure

  if metric.startswith('bleu'):
      bleu = load_metric("bleu", seed = 0)
      for c in cols:
          bleu_tokspred = [x.split() for x in generated_data[c].to_list()]
          bleu_preds= [[x.translate(str.maketrans('', '', string.punctuation)).lower() for x in sublist] for sublist in bleu_tokspred]
          bleu_toksref=[x.split() for x in generated_data[reference_column].to_list()]
          bleu_refs= [[[x.translate(str.maketrans('', '', string.punctuation)).lower() for x in sublist]] for sublist in bleu_toksref]
          for i in range(len(generated_data)):
              if len(bleu_preds[i]) == 0:
                  data[c][i] = 0
              else:
                  bleu.add(prediction=bleu_preds[i], reference=bleu_refs[i])
                  if metric == 'bleu-1':
                      bleu_output = bleu.compute(max_order=1)
                  if metric == 'bleu-3':
                      bleu_output = bleu.compute(max_order=3)
                  if metric == 'bleu-4':
                      bleu_output = bleu.compute(max_order=4)
                  data[c][i] = bleu_output['bleu']
  data['metric'] = metric
  return data

def unanimous(seq):
  """
  check if two sequences are identical, returns boolean
  """
  it = iter(seq)
  try:
    first = next(it)
  except StopIteration:
    return True
  else:
    return all(i == first for i in it)

def rank_CNs(input_data, index_column = 'index'):
  """
  Rank generated CNs evaluated by evaluator function
  """
  ranked_dfs = []
  df = input_data.copy()
  df_filtered = df[[col for col in df.columns if col not in [index_column, 'metric']]]
  # I rank the scores obtained with the same metric by all the CNs of the same row
  # a higher ranking (== number) is assigned to a CN with higher score
  ranked = df_filtered.rank(axis=1)
  ranked[index_column] = df[index_column]
  ranked['metric'] = df['metric']
  return ranked


def get_evaluated_and_ranked_dfs(generated_data, metrics = ['rouge', 'bleu-1', 'bleu-3', 'bleu-4']):
  """
  Return two dataframes which are the concatenation of the generated data (i) evaluated and (ii) ranked with different metrics
  """
  ranked_dfs = []
  evaluated_dfs = []
  for m in metrics:
    evaluated_d = evaluator(generated_data, metric=m)
    ranked_d = rank_CNs(evaluated_d)
    evaluated_dfs.append(evaluated_d)
    ranked_dfs.append(ranked_d)
  evaluated = pd.concat(evaluated_dfs).reset_index(drop=True)
  ranked = pd.concat(ranked_dfs).reset_index(drop=True)
  return evaluated, ranked


def get_avg_rank(ranked, index_column = 'index'):
  """
  Returns a dataset containing the average of the rankings obtained by each generation with the various metric, and a column
  containing the configuration(s) achieving the highest (i.e. the best) average rank.
  """
  avg_rank = ranked.groupby('index').mean()
  f2 = lambda r: ','.join(avg_rank.columns[r])
  # choose the CN with highest mean rank: if ties, choose them both with ',' as separator (result is put in 'eq_' column)
  avg_rank['best_rank']=avg_rank.eq(avg_rank.max(axis=1), axis=0).apply(f2, axis=1)
  avg_rank = avg_rank.reset_index()
  return avg_rank


def select_among_ties(avg_rank, ranked, generated_data, index_column = 'index'):
  """
  Adds a column to the dataframe eith average rank, indicating which configuration is selected in case of ties.
  selects (i) the generation achieving a higher ranking at least once (ii) otherwise, if the rankings are identical, choose randomly
  """

  #I create a subset comprising all the rows with a tie
  avg_rank['tie'] = 0
  #I create the column selected as a copy of the column 'best_rank'
  avg_rank['selected_config'] = avg_rank['best_rank']

  ties_df = avg_rank[avg_rank['best_rank'].str.contains(",", na=False)].reset_index(drop=True)
  for i in ties_df[index_column]:
    cns = {}
    ties = {}
    thisrowties = ties_df.loc[ties_df[index_column]==i,'best_rank'].values[0].split(',')

    for el in thisrowties:
      #I append to a dictionary the ties and their rankings
      ties[el]= sorted(ranked.loc[ranked[index_column]==i, el].values.flatten().tolist())
      #I append to a dictionary the ties and their CN
      cns[el] = generated_data.loc[generated_data[index_column]==i,el].values[0]

    # I check if there are identical rankings for the CNs with the tie
    if unanimous(ties.values()):
      # if they have equal rankings I set '1' as value in the '_tie' column
      avg_rank.loc[avg_rank[index_column]==i,'tie']=1
      #I choose randomly the selected CN
      r = random.Random(0)
      selected = r.choice(thisrowties)
      avg_rank.loc[avg_rank[index_column]==i, 'selected_config'] = selected
    else:
      # if they have different rankings, I choose the one achieving the highest ranking at least once
      selected = max(ties.items(), key=operator.itemgetter(1))[0]
      avg_rank.loc[avg_rank[index_column]==i, 'selected_config'] = selected

  return avg_rank


def get_out_data(generated_data, avg_rank, evaluated, index_column = 'index', reference_column = 'CN_ed', metrics = ['rouge', 'bleu-1', 'bleu-3', 'bleu-4']):
  """
  Returns a dataframe with index column, reference column, selected CN and configuration, and the scores obtained with the various metrics
  """
  out_data = generated_data[[index_column, reference_column]]
  out_data['selected_CN'] = np.nan
  out_data['selected_config'] = avg_rank['selected_config']
  out_data[[m for m in metrics]] = np.nan

  for i in avg_rank[index_column]:
    selected_config = out_data.loc[out_data[index_column] == i, 'selected_config'].values[0]
    out_data.loc[out_data[index_column]==i, 'selected_CN'] = generated_data.loc[generated_data[index_column]==i, selected_config].values[0]
    for m in metrics:
      out_data.loc[out_data[index_column]==i, m] = evaluated.loc[(evaluated[index_column]==i) & (evaluated['metric']==m), selected_config].values[0]
  return out_data

def create_best_dataset(generated_data, ranked, evaluated, list_of_elements, index_column = 'index', reference_column = 'CN_ed'):
  """
  Takes in input generated, ranked, evaluated data and the list of (i) models (ii) decodings (iii) model-decoding combinations which corresponds to the columns names
  and returns (i) best model dataset (ii) best decoding dataset (iii) best model decoding dataset
  """
  list_of_dfs = []
  for el in list_of_elements:
    generations_ = generated_data[[c for c in generated_data.columns if el in c or c in [reference_column, index_column]]]
    ranked_ = ranked[[c for c in ranked.columns if el in c or c in [index_column, 'metric']]]
    evaluated_ = evaluated[[c for c in evaluated.columns if el in c or c in [index_column, 'metric']]]

    avg_rank_df = get_avg_rank(ranked_)
    avg_rank_df = select_among_ties(avg_rank_df, ranked_, generations_)
    df_ = get_out_data(generations_, avg_rank_df, evaluated_)

    list_of_dfs.append(df_)

  return pd.concat(list_of_dfs).reset_index(drop=True)
  
  
def walk_tree(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth

def get_max_sd(input_text):
  """
  Get the Maximum Syntactic Depth (MSD): the maximum depth among the dependency trees calculated over each sentence composing a text.
  """
  docu = en_nlp(input_text)
  return max([walk_tree(sent.root, 0) for sent in docu.sents])

def get_avg_sd(input_text):
  """
  Get the average depth of the sentences in each text.
  """
  docu = en_nlp(input_text)
  return statistics.mean([walk_tree(sent.root, 0) for sent in docu.sents])

def get_nst(input_text):
  """
  Get the number of sentences in the input text.
  """
  docu = en_nlp(input_text)
  return len([sent for sent in docu.sents])