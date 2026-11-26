# SPDX-License-Identifier: Apache-2.0
# Sliding-window dataset inspired by Sebastian Raschka (see NOTICE).
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, token_ids, max_length, stride):
        if max_length < 1 or stride < 1:
            raise ValueError('Length and stride must be positive')
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.starts = range(0, len(token_ids) - max_length, stride)
        self.max_length = max_length

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        i = self.starts[index]
        return self.tokens[i:i+self.max_length], self.tokens[i+1:i+self.max_length+1]


def make_loaders(path, context_length=64, batch_size=4):
    import tiktoken
    text = Path(path).read_text(encoding='utf-8')
    ids = tiktoken.get_encoding('gpt2').encode(text, disallowed_special=())
    split = int(len(ids) * 0.9)
    train = GPTDataset(ids[:split], context_length, context_length)
    val = GPTDataset(ids[split:], context_length, context_length)
    if len(train) < batch_size or not len(val):
        raise ValueError('Corpus too short for this context length and batch size')
    print(f'Tokens: {len(ids)} | Train: {split} | Validation: {len(ids)-split}')
    return (DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/w_pustyni_i_w_puszczy.txt')
    args = p.parse_args()
    a, b = make_loaders(args.data)
    print(f'Train batches: {len(a)} | Validation batches: {len(b)}')
