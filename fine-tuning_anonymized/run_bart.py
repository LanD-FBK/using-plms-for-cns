from datasets import load_dataset
import numpy as np
import transformers
import datasets
import datetime
import os
import pandas
from datasets import Dataset
import logging
from transformers import set_seed
logger = logging.getLogger(__name__)

seed = 21
set_seed(seed)

os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'

df = pandas.read_csv(path_to_hscn)
train_df = df[df.Assigned_to == 'train']
val_df = df[df.Assigned_to == 'dev']
test_df = df[df.Assigned_to == 'test']

train_dataset = Dataset.from_pandas(train_df,  split='train').shuffle(seed=seed) #let's shuffle the trainset
valid_dataset = Dataset.from_pandas(val_df,  split='validation')
test_dataset = Dataset.from_pandas(test_df,  split='test')

print(train_dataset.num_rows)
print(test_dataset.num_rows)
print(valid_dataset.num_rows)

it = iter(train_dataset)
data = next(it)
print("Example data from the dataset: \n", data['HS_ed'])
data = next(it)
print("Example data from the dataset: \n", data)


#yes
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
#from transformers.modeling_bart import shift_tokens_right
import torch

batch_size = 4
max_len = 128
model_path = 'bart-base'
model_name = 'facebook/bart-base' # 'facebook/bart-large'
tokenizer = BartTokenizer.from_pretrained(model_name)

def model_init():
    return BartForConditionalGeneration.from_pretrained(model_name)

# they suggest to use distilbart-xsum-12-6 for finetuning with our own data

#https://github.com/huggingface/transformers/issues/8005
inputs = tokenizer.prepare_seq2seq_batch(
    src_texts=train_dataset['HS_ed'],
    tgt_texts=train_dataset['CN_ed'],
    max_length=max_len, max_target_length=max_len,
    return_tensors='pt'
)
# This function is copied from modeling_bart.py
def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


tgt = inputs['labels'][0].tolist()
decoder_inputs = shift_tokens_right(inputs['labels'], tokenizer.pad_token_id)[0].tolist()
print("decoder inputs:", tokenizer.decode(decoder_inputs))
print("target:", tokenizer.decode(tgt))


def encode(example_batch):
  input_encodings = tokenizer.batch_encode_plus(example_batch['HS_ed'], padding="max_length", max_length=max_len, truncation=True) #padding='longest' makes various sizes inside batch
  target_encodings = tokenizer.batch_encode_plus(example_batch['CN_ed'], padding="max_length", max_length=max_len, truncation=True)
  labels = target_encodings['input_ids']
  tlabels = torch.Tensor(labels).to(torch.int64)
  decoder_input_ids = shift_tokens_right(tlabels, tokenizer.pad_token_id)
  #model.config.pad_token_id
  tlabels[tlabels[:, :] == tokenizer.pad_token_id] = -100
  encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': decoder_input_ids.numpy(),
        'labels': tlabels.numpy(),
    }
  return encodings


t_dataset = train_dataset.map(encode, batched=True)
v_dataset = valid_dataset.map(encode, batched=True)
columns = ['input_ids', 'labels', 'decoder_input_ids','attention_mask',]
t_dataset.set_format(type='torch', columns=columns)
v_dataset.set_format(type='torch', columns=columns)

def cn_hp_space(trial):

    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]), #trial.suggest_float("learning_rate", 1e-6, 1e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5, log=True),
        "warmup_ratio":trial.suggest_categorical("warmup_ratio", [0, 0.1])
        #"weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
        #"adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True)
    }


from transformers import EarlyStoppingCallback
batch_size = 4
#study freeze_params at some point.

training_args = TrainingArguments(
    output_dir=model_path,
    #num_train_epochs=3,       #1
    #per_device_train_batch_size=batch_size, #1
    per_device_eval_batch_size=batch_size,   #1
    #warmup_steps=500, #    maybe we can use a percentage, like 10%
    #weight_decay=0.01,
    logging_dir='./logs',
    #added by me
    #learning_rate=1e-5, #default is 5e-5
    eval_steps=500, #default is also 500
    #load_best_model_at_end=True,
    #save_steps=1000,
    save_total_limit=4,
    evaluation_strategy='steps',

)

trainer = Trainer(
    #model=model,
    model_init=model_init,
    args=training_args,
    train_dataset=t_dataset,
    eval_dataset=v_dataset,
)

best_run = trainer.hyperparameter_search(n_trials=10, direction="minimize", hp_space=cn_hp_space)


# trainer.add_callback(EarlyStoppingCallback(1, 0.0001)) # we can change the criteria, in transformers test file it is 0.0001
# when it is 0.001, it stops too early at the 1500th step.

# print("Trainable Params:",sum(p.numel() for p in model.parameters() if p.requires_grad))
# print("Total Params", sum(p.numel() for p in model.parameters()))


# trainer.train()

import math

logger.info("*** Evaluate ***")

eval_output = trainer.evaluate()

results = {}
output_dir = ""
output_eval_file = os.path.join(output_dir, "bart_eval_results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results *****")
    for key in sorted(eval_output.keys()):
        logger.info("  %s = %s", key, str(eval_output[key]))
        writer.write("%s = %s\n" % (key, str(eval_output[key])))
    writer.write(f"Perplexity: {math.exp(eval_output['eval_loss']):.2f}")

results.update(eval_output)

