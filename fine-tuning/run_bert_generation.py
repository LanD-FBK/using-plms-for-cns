import torch
import pandas
from datasets import Dataset
import datasets
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List
import pandas

os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'

from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
import transformers
transformers.logging.set_verbosity_info()

# import gc
# gc.collect()
# torch.cuda.empty_cache()

logger = logging.getLogger(__name__)
seed = 21
set_seed(seed)


from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
print(tokenizer.vocab_size)
df = pandas.read_csv(path_to_hscn)
train_df = df[df.Assigned_to == 'train']
trn_df = train_df[['HS_ed','CN_ed']]
val_df = df[df.Assigned_to == 'dev']
v_df = val_df[['HS_ed','CN_ed']]
test_df = df[df.Assigned_to == 'test']
tst_df = test_df[['HS_ed','CN_ed']]


train_data = Dataset.from_pandas(trn_df,  split='train').shuffle(seed=seed)
val_data = Dataset.from_pandas(v_df,  split='validation')
#test_data = Dataset.from_pandas(tst_df,  split='test')

batch_size = 4 # change to 16 for full training
encoder_max_length = 92 #max hs len
decoder_max_length = 128 #max cn len

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["HS_ed"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["CN_ed"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch


train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["HS_ed", "CN_ed"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["HS_ed", "CN_ed"]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

from transformers import EncoderDecoderModel


def model_init():
    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")#, tie_encoder_decoder=True)

    # set special tokens
    bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
    bert2bert.config.eos_token_id = tokenizer.eos_token_id
    bert2bert.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
    bert2bert.config.max_length = 128 #142
    bert2bert.config.min_length = 25 #56
    bert2bert.config.no_repeat_ngram_size = 3
    bert2bert.config.early_stopping = True
    bert2bert.config.length_penalty = 2.0
    bert2bert.config.num_beams = 4

    return bert2bert

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback


# load rouge for validation
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

def cn_hp_space(trial):

    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5,2e-5,3e-5,4e-5, 5e-5]), #trial.suggest_float("learning_rate", 1e-6, 1e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2, 4]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5, log=True),
        "warmup_ratio":trial.suggest_categorical("warmup_ratio", [0, 0.1])
        #"weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
        #"adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True)
    }


# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="bert-base",
    #per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size*2,
    predict_with_generate=True,
    #evaluate_during_training=True,
    do_train=True,
    do_eval=True,
    #num_train_epochs=5, #added by Serra also add later weight_decay=0.01,
    #weight_decay=0.01,
    logging_steps=500,  # set to 1000 for full training
    save_steps=500,  # set to 500 for full training
    eval_steps=500,  # set to 8000 for full training
    #warmup_steps=500,  # set to 2000 for full training
    #learning_rate=2e-5,
    #max_steps=16, # delete for full training
    overwrite_output_dir=True,
    save_total_limit=4,
    evaluation_strategy='steps',
    logging_dir='./logs'
    #load_best_model_at_end=True,
    #fp16=True,
)

# bert2bert.zero_grad()

# instantiate trainer
trainer = Seq2SeqTrainer(
    #model=bert2bert,
    model_init=model_init,
    tokenizer=tokenizer,
    args=training_args,
    #compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,

)

# trainer.add_callback(EarlyStoppingCallback(1, 0.0001))

# trainer.train()

# print("Trainable Params:",sum(p.numel() for p in bert2bert.parameters() if p.requires_grad))
# print("Total Params", sum(p.numel() for p in bert2bert.parameters()))

best_run = trainer.hyperparameter_search(n_trials=10, direction="minimize", hp_space=cn_hp_space)


import math

logger.info("*** Evaluate ***")

eval_output = trainer.evaluate()

results = {}
output_dir = ""
output_eval_file = os.path.join(output_dir, "bert_eval_results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results *****")
    for key in sorted(eval_output.keys()):
        logger.info("  %s = %s", key, str(eval_output[key]))
        writer.write("%s = %s\n" % (key, str(eval_output[key])))
    writer.write(f"Perplexity: {math.exp(eval_output['eval_loss']):.2f}")

results.update(eval_output)

