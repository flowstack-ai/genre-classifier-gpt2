import torch
from torch.utils.data import Dataset
from ftfy import fix_text


class MovieGenresDataset(Dataset):
    def __init__(self, path):
        self.dataset = open(path, 'r', encoding='utf-8')
        self.dataset = self.dataset.read().strip().split('\n')
        self.output_labels = {}
        self.labels = []
        self.texts = []
        output_label_id = 0
        for i in range(len(self.dataset)):
            delimitted = self.dataset[i].split(':::')
            self.labels.append(delimitted[2].strip())
            self.texts.append(fix_text(delimitted[3].strip()))
            if delimitted[2].strip() not in self.output_labels:
                self.output_labels[delimitted[2].strip()] = output_label_id
                output_label_id += 1
        self.n_examples = len(self.labels)
        self.n_output_labels = len(self.output_labels)
    
    def __len__(self):
        return self.n_examples

    def __getitem__(self, index):
        return { 
            'text':self.texts[index], 
            'label':self.labels[index]
        }


class GPT2ClassificationCollate(object):
    def __init__(self, tokenizer, labels, max_seq_len):
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        labels = [self.labels[label] for label in labels]
        inputs = self.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_seq_len)
        inputs.update({'labels':torch.tensor(labels)})
        return inputs