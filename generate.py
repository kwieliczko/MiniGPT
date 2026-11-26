# SPDX-License-Identifier: Apache-2.0
import argparse
import torch
from model import MiniGPT


@torch.no_grad()
def generate_ids(model, ids, max_new_tokens):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(ids[:, -model.context_length:])
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, next_id), dim=1)
    return ids


def generate(checkpoint, prompt, max_new_tokens=40):
    import tiktoken
    if not prompt or max_new_tokens < 0:
        raise ValueError('Nonempty prompt and nonnegative token count required')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = MiniGPT(**saved['config'])
    model.load_state_dict(saved['model_state_dict'])
    tokenizer = tiktoken.get_encoding('gpt2')
    ids = torch.tensor([tokenizer.encode(prompt, disallowed_special=())])
    return tokenizer.decode(generate_ids(model, ids, max_new_tokens)[0].tolist())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/minigpt_checkpoint.pth')
    p.add_argument('--prompt', required=True)
    p.add_argument('--max-new-tokens', type=int, default=40)
    print(generate(**vars(p.parse_args())))
