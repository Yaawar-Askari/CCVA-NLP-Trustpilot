import random
import numpy as np
from itertools import cycle
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import pandas as pd


class TrustpilotPaired(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128, train=True, transform=None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train
        self.transform = transform

        # Split the data into training and validation sets
        train_data, val_data = train_test_split(self.data, test_size=0.2, random_state=42)
        self.data = train_data if train else val_data

        self.data_dict = {}

        for i in range(len(self.data)):
            review = self.data.iloc[i]['review']
            label = self.data.iloc[i]['sentiment']

            if label not in self.data_dict:
                self.data_dict[label] = []
            self.data_dict[label].append(review)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        review = self.data.iloc[index]['review']
        label = self.data.iloc[index]['sentiment']

        # Tokenize the review
        inputs = self.tokenizer(review, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # Randomly select another review of the same sentiment
        paired_review = random.SystemRandom().choice(self.data_dict[label])
        paired_inputs = self.tokenizer(paired_review, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        paired_input_ids = paired_inputs['input_ids'].squeeze()
        paired_attention_mask = paired_inputs['attention_mask'].squeeze()

        return input_ids, attention_mask, paired_input_ids, paired_attention_mask, label


if __name__ == '__main__':
    """
    Test code for the data loader
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    trustpilot_paired = TrustpilotPaired(csv_file='path/to/trustpilot.csv', tokenizer=tokenizer)
    loader = cycle(DataLoader(trustpilot_paired, batch_size=16, shuffle=True, num_workers=0, drop_last=True))

    print(trustpilot_paired.data_dict.keys())

    input_ids_batch, attention_mask_batch, paired_input_ids_batch, paired_attention_mask_batch, labels_batch = next(loader)
    print(labels_batch)

    # Since this is text data, visualization would typically involve text processing or inspection
    print(tokenizer.decode(input_ids_batch[0]))
    print(tokenizer.decode(paired_input_ids_batch[0]))
