# MiniGPT — From Tokens to Language

An educational GPT-style language model trained from scratch on Polish literature.

**Version 0.1.0 — experimental, pre-training release.** No trained weights or
measured full-corpus results are included. Companion blog: https://www.thinkingaboutdata.com/

## Architecture

GPT-2 BPE tokenizer (50,257 tokens), embedding dimension 64, context 64 tokens,
2 transformer blocks, 4 attention heads per block, 6,537,088 trainable parameters.
The model uses causal self-attention, pre-normalization, GELU and residual connections.
It does not load pretrained GPT-2 weights. It is not a chatbot.

## Quick start

Python 3.10+ recommended. Create an isolated environment and install dependencies:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For NVIDIA GPU training, install a PyTorch build compatible with your CUDA setup
(https://pytorch.org/get-started/locally/). CPU training is also supported.
Place the UTF-8 novel in `data/w_pustyni_i_w_puszczy.txt` (see data/README.md).

```sh
python prepare_data.py
python train.py --epochs 5 --batch-size 4
python generate.py --prompt "Staś zobaczył Nel" --max-new-tokens 40
```

Alternatively, launch `jupyter lab` from the repository root and open `MiniGPT.ipynb`.

Training splits token IDs into contiguous 90% training and 10% validation sections
before constructing non-overlapping 64-token windows with shifted targets.
AdamW uses learning rate 0.0003 by default. Training loss is averaged during each
epoch; validation loss is calculated at epoch end. The final checkpoint contains
model and optimizer states and loss histories, not an exact-resume RNG snapshot.
Generation uses greedy decoding and a rolling context window; repetition is possible.
A single novel is insufficient for broad generalization. Runtime and quality depend
on the hardware, corpus and training configuration. Seed 123 does not guarantee
bitwise reproducibility across platforms.

## Attribution and license

The main educational reference is Sebastian Raschka's *Build a Large Language Model
(From Scratch)* (Manning, 2024). Official source code:
https://github.com/rasbt/LLMs-from-scratch

The GPT-2 architecture and tokenizer originate from OpenAI:
https://github.com/openai/gpt-2

MiniGPT is a simplified, AI-assisted educational implementation. It is not affiliated
with or endorsed by Sebastian Raschka, Manning or OpenAI. See `NOTICE` for details.
The source code of this repository is offered under Apache License 2.0, subject to
applicable third-party rights. The license does not extend to third-party books,
illustrations, datasets or dependencies. See `LICENSE`. No novel text is bundled.
