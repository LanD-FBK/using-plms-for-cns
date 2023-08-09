import torch
import pandas
from datasets import Dataset
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List


from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)

logger = logging.getLogger(__name__)

os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'

seed = 21
set_seed(seed)

model_name = 't5-base'

tokenizer = T5Tokenizer.from_pretrained(model_name)

# process the examples in input and target text format and the eos token at the end
def add_eos_to_examples(example):
    example['input_text'] = '%s </s>' % example['HS_ed']
    example['target_text'] = '%s </s>' % example['CN_ed']
    return example

# tokenize the examples
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], padding="max_length", max_length=128, truncation=True)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'],padding="max_length", max_length=128, truncation=True)
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings


def load_data():
    df = pandas.read_csv(path_to_hscn)
    train_df = df[df.Assigned_to == 'train']
    val_df = df[df.Assigned_to == 'dev']
    test_df = df[df.Assigned_to == 'test']

    train_dataset = Dataset.from_pandas(train_df,  split='train').shuffle(seed=seed)
    valid_dataset = Dataset.from_pandas(val_df,  split='validation')
    test_dataset = Dataset.from_pandas(test_df,  split='test')

    # map add_eos_to_examples function to the dataset example wise
    train_dataset = train_dataset.map(add_eos_to_examples)
    # map convert_to_features batch wise
    train_dataset = train_dataset.map(convert_to_features, batched=True)

    valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
    valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

    # set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    torch.save(train_dataset, 'train_data.pt')
    torch.save(valid_dataset, 'valid_data.pt')



# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class T2TDataCollator:  # (DataCollator)
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:  # collate_batch
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        labels = torch.stack([example['target_ids'] for example in batch])
        labels[labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'decoder_attention_mask': decoder_attention_mask
        }

load_data()

def cn_hp_space(trial):

    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [2e-5,3e-5,4e-5]),#[1e-5, 1e-3, 1e-4, 5e-4, 5e-5]), #trial.suggest_float("learning_rate", 1e-6, 1e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2, 4]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5, log=True),
        "warmup_ratio":trial.suggest_categorical("warmup_ratio", [0, 0.1])
        #"weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
        #"adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True)
    }

batch_size = 4
#study freeze_params at some point.
learning_rate = 0.001 #in original paper 0.001 ,  not great I suppose: 1e-4
train_dataset  = torch.load('train_data.pt')
valid_dataset = torch.load('valid_data.pt')
model_path = 't5_models_base/'
#model = T5ForConditionalGeneration.from_pretrained(model_name)

def model_init():
    return T5ForConditionalGeneration.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir=model_path,
    num_train_epochs=3,       #1
    per_device_train_batch_size=batch_size, #1
    per_device_eval_batch_size=batch_size,   #1
    warmup_steps=300, #    maybe we can use a percentage, like 10%
    #weight_decay=0.01,
    logging_dir='./logs',
    #added by me
    learning_rate=learning_rate, #default is 5e-5
    eval_steps=500, #default is also 500
    load_best_model_at_end=True,
    save_steps=1000,
    save_total_limit=4,
    evaluation_strategy='steps',
    remove_unused_columns=False,

)

trainer = Trainer(
    #model=model,
    model_init = model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=T2TDataCollator(),

)
#best_run = trainer.hyperparameter_search(n_trials=10, direction="minimize", hp_space=cn_hp_space)

trainer.add_callback(EarlyStoppingCallback(1, 0.0001))
# we can change the criteria, in transformers test file it is 0.0001
#when it is 0.001, it stops too early at the 1500th step.

trainer.train()

# print("Trainable Params:",sum(p.numel() for p in model.parameters() if p.requires_grad))
# print("Total Params", sum(p.numel() for p in model.parameters()))

import math

logger.info("*** Evaluate ***")

eval_output = trainer.evaluate()

results = {}
output_dir = ""
output_eval_file = os.path.join(output_dir, "t5_eval_results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results *****")
    for key in sorted(eval_output.keys()):
        logger.info("  %s = %s", key, str(eval_output[key]))
        writer.write("%s = %s\n" % (key, str(eval_output[key])))
    writer.write(f"Perplexity: {math.exp(eval_output['eval_loss']):.2f}")

results.update(eval_output)





