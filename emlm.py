from typing import final
from datasets import load_dataset
from sklearn.utils import shuffle
import nltk.data
import pandas as pd
from tqdm import tqdm
import os
import argparse

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForWholeWordMask, DataCollatorForLanguageModeling
from transformers.models.bert.configuration_bert import BertConfig
from transformers.utils.dummy_pt_objects import AutoModel
from datasets import interleave_datasets

from emlm_data_collator import DataCollatorForEMLM

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_file", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--emolex_path", type=str)
parser.add_argument("--from_scratch", type=int)
parser.add_argument("--k", type=float)
parser.add_argument("--testing_epochs", type=int)
parser.add_argument("--max_seq_length", type=int, default=512, help="Default 512")

word_key = 'English (en)'
emotions = [
    'Positive', 'Negative', 'Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy',
    'Sadness', 'Surprise', 'Trust'
]

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nltk.download('punkt')


def give_generator(dataset, sentence_tokenizer):
    def generate():
        crt_idx = 0
        underlying_dataset = dataset

        while (True):
            if crt_idx == len(dataset):
                crt_idx = 0

            for sentence in sentence_tokenizer.tokenize(
                    dataset[crt_idx]['text']):
                yield sentence

            crt_idx += 1

    return generate


class EMLMDataset(torch.utils.data.IterableDataset):
    def __init__(self, underlying_dataset, generator_function,
                 sentence_tokenizer) -> None:
        super(EMLMDataset).__init__()
        self.generator_function = give_generator(underlying_dataset,
                                                 sentence_tokenizer)

    def __iter__(self):
        return iter(self.generator_function())


def main():

    emotion_set = set()
    df = pd.read_excel(args.emolex_path)

    for i, row in df.iterrows():
        for emo in emotions:
            if row[emo] == 1:
                emotion_set.add(row[word_key])

    if len(os.listdir(args.checkpoint_file)) == 0:
        if args.from_scratch == 1:
            bertconfig = BertConfig()
            bert_model = BertForMaskedLM(bertconfig)
            with open('logs.txt', 'a') as f:
                f.write("Intantiating for the FIRST TIME from scratch\n")
        else:
            bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            with open('logs.txt', 'a') as f:
                f.write(
                    "Intantiating for the FIRST TIME from base uncased checkpoint\n")
    else:
        bert_model = BertForMaskedLM.from_pretrained(args.checkpoint_file)
        with open('logs.txt', 'a') as f:
            f.write("Intantiating from an already saved checkpoint\n")

    wikipedia_dataset = load_dataset("wikipedia", "20200501.en")
    wikipedia_dataset = wikipedia_dataset.remove_columns("title")
    bookcorpus_dataset = load_dataset("bookcorpus")

    final_dataset = interleave_datasets(
        [wikipedia_dataset['train'], bookcorpus_dataset['train']])

    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    dataset = EMLMDataset(final_dataset, give_generator, sentence_tokenizer)
    dataset = dataset.shuffle(buffer_size=256)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size)

    data_collator = DataCollatorForEMLM(tokenizer=bert_tokenizer,
                                        mlm=True,
                                        mlm_probability=0.15)
    data_collator.k = args.k
    data_collator.emotion_set = emotion_set

    bert_model.to(device)

    optimizer = torch.optim.Adam(bert_model.parameters(),
                                 lr=1e-4,
                                 weight_decay=0.01,
                                 betas=[0.9, 0.999])
    grad_acc = 512 // args.batch_size
    optimizer.zero_grad()

    crt_loss = 0
    while (True):
        for i, elem in enumerate(dataloader):
            if i % 50000 == 0:
                print('*****Evaluating*****')
                # Save model to free up space for downstream training
                bert_model.save_pretrained(args.checkpoint_file)

                # ----- You can train on downstream tasks here to see how the 
                # model behaves -----

                bert_model = BertForMaskedLM.from_pretrained(
                    args.checkpoint_file)
                bert_model.to(device)
                optimizer = torch.optim.Adam(bert_model.parameters(),
                                             lr=1e-4,
                                             weight_decay=0.01,
                                             betas=[0.9, 0.999])

            tokenized = bert_tokenizer(elem)["input_ids"]
            inp = data_collator(tokenized)

            if inp['input_ids'].shape[1] > args.max_seq_length:
                continue

            inp = {k: inp[k].to(device) for k in inp}

            result = bert_model(inp["input_ids"], labels=inp["labels"])
            crt_loss += result.loss.cpu().detach().numpy()

            result.loss.backward()
            if i % grad_acc == 0 and i != 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % (grad_acc * 50) == 0 and i != 0:
                print("Current Step", i, flush=True)
                print("Current Loss", crt_loss / (grad_acc * 50), flush=True)
                with open('results_performance.txt', 'a') as f:
                    f.write('Average loss the past ' + str(grad_acc * 50) +
                            " steps: " + str(crt_loss /
                                             (grad_acc * 50)) + '\n')
                crt_loss = 0


if __name__ == "__main__":
    main()
