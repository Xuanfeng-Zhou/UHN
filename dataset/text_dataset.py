import os
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import csv
import torch
import re
from tqdm import tqdm

UNK, PAD, CLS = "<unk>", "<pad>", "<cls>"
PAD_TOKEN_IDX = 1

# Tokenizer
_BASIC_ENGLISH_RE = re.compile(r"\w+\'\w+|\w+|[^\w\s]", re.UNICODE)
def tokenize(text):
    return _BASIC_ENGLISH_RE.findall(text.lower())

def build_vocab(dataset_name, root, tokenizer, max_size):
    specials = [UNK, PAD, CLS] 
    if dataset_name == 'imdb':
        counter = Counter()
        data_dir = os.path.join(root, 'imdb', 'train')
        for label_dir in os.listdir(data_dir):
            if label_dir == 'unsup':
                continue
            full_dir = os.path.join(data_dir, label_dir)
            if os.path.isdir(full_dir):
                for fname in os.listdir(full_dir):
                    with open(os.path.join(full_dir, fname), 'r', encoding='utf-8') as f:
                        tokens = tokenizer(f.read().strip())
                        counter.update(tokens)
        # Include specials in the beginning of the vocab
        most_common = counter.most_common(max_size - len(specials))
        vocab = {word: idx for idx, (word, _) in enumerate(most_common, start=len(specials))}
        for idx, token in enumerate(specials):
            vocab[token] = idx
        return vocab
    elif dataset_name == 'ag_news':
        counter = Counter()
        # Read the CSV and tokenize text
        data_dir = os.path.join(root, 'ag_news', 'train.csv')
        with open(data_dir, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                # Assuming the format is: label, title, content
                _, title, content = line
                tokens = tokenizer(f"{title} {content}")
                counter.update(tokens)
        
        # Include special tokens at the beginning of the vocab
        most_common = counter.most_common(max_size - len(specials))
        vocab = {word: idx for idx, (word, _) in enumerate(most_common, start=len(specials))}
        
        for idx, token in enumerate(specials):
            vocab[token] = idx
        
        return vocab    
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

# Custom Dataset
class IMDBDataset(Dataset):
    def __init__(self, root, vocab, max_seq_len, train):
        label_map = {"neg": 0, "pos": 1}
        self.labels = []
        self.token_texts_ids = []        
        
        data_dir = os.path.join(root, 
                                'imdb',
                                'train' if train else 'test')
        for label_name, label_id in label_map.items():
            label_dir = os.path.join(data_dir, label_name)
            for fname in tqdm(os.listdir(label_dir), 
                    desc=f"Loading [{'train' if train else 'test'}] {label_name} files"):
                self.labels.append(label_id)
                with open(os.path.join(label_dir, fname), 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                tokens = tokenize(text)
                token_ids = [vocab[CLS]] + [vocab.get(token, vocab[UNK]) for token in tokens[:max_seq_len - 1]]
                self.token_texts_ids.append(torch.tensor(token_ids, dtype=torch.long))
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        token_ids = self.token_texts_ids[index]
        return token_ids, label

class AGNewsDataset(Dataset):
    def __init__(self, root, vocab, max_seq_len, train):
        self.label_map = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}
        self.labels = []
        self.token_texts_ids = []

        csv_path = os.path.join(root, 
                                'ag_news',
                                'train.csv' if train else 'test.csv')
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip the header
            next(reader)
            for line in tqdm(reader, 
                    desc=f"Loading [{'train' if train else 'test'}] AG News data"):
                label, title, content = line
                label = int(label) - 1  # Assuming AG News labels are 1-based
                text = f"{title} {content}"
                self.labels.append(label)
                tokens = tokenize(text)
                token_ids = [vocab[CLS]] + [vocab.get(token, vocab[UNK]) for token in tokens[:max_seq_len - 1]]
                self.token_texts_ids.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        token_ids = self.token_texts_ids[index]
        return token_ids, label
    
# Collate function
def collate_text_batch(batch, vocab):
    texts, labels = zip(*batch)
    return pad_sequence(texts, batch_first=True, padding_value=vocab[PAD]), \
        torch.tensor(labels, dtype=torch.long)
