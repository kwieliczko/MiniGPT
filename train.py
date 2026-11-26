# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path
import time
import hashlib
import os
from generate import generate_ids
import torch
import torch.nn.functional as F
from config import MODEL_CONFIG, SEED
from model import MiniGPT
from prepare_data import make_loaders


def evaluate_model(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum').item()
            total_tokens += y.numel()
    return total_loss / total_tokens


def save_atomic(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def train(data, epochs=5, batch_size=4, lr=0.0003, output='checkpoints', device=None, resume=False):
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError('Epochs, batch size and learning rate must be positive')
    torch.manual_seed(SEED)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    train_loader, val_loader = make_loaders(data, MODEL_CONFIG['context_length'], batch_size)
    model = MiniGPT(**MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print('Device:', device, '| Parameters:', sum(p.numel() for p in model.parameters()))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'minigpt_checkpoint.pth'
    settings = dict(batch_size=batch_size, lr=lr,
                    corpus_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest())
    completed = 0
    train_losses, val_losses, epoch_seconds, samples = [], [], [], []
    best = float('inf')
    if resume:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('settings') != settings or saved['config'] != MODEL_CONFIG:
            raise ValueError('Checkpoint settings differ or checkpoint predates v0.2.0')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        completed = saved['epochs']
        initial = saved['initial_val_loss']
        train_losses = saved['train_losses']
        val_losses = saved['val_losses']
        epoch_seconds = saved['epoch_seconds']
        samples = saved['samples']
        best = saved['best_val_loss']
        torch.set_rng_state(saved['rng_state'])
        if device.type == 'cuda' and saved['cuda_rng_state'] is not None:
            if len(saved['cuda_rng_state']) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
        print(f'Resuming after epoch {completed}')
    else:
        if path.exists():
            raise FileExistsError('Checkpoint exists; use resume=True or a new output directory')
        initial = evaluate_model(model, val_loader, device)
        print(f'Initial validation loss: {initial:.4f}')
    import tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')
    start = time.perf_counter()
    for epoch in range(completed, completed + epochs):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, total_tokens = 0.0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            if step % 100 == 0:
                print(f'Epoch {epoch+1}, step {step}/{len(train_loader)}: {loss.item():.4f}')
        train_losses.append(total_loss / total_tokens)
        val_losses.append(evaluate_model(model, val_loader, device))
        epoch_seconds.append(time.perf_counter() - epoch_start)
        print(f'Epoch {epoch+1}/{completed+epochs} | Train loss: {train_losses[-1]:.4f} | Validation loss: {val_losses[-1]:.4f}')
        prompt = torch.tensor([tokenizer.encode('Staś zobaczył Nel')], device=device)
        sample = tokenizer.decode(generate_ids(model, prompt, 40)[0].tolist())
        samples.append(sample)
        improved = val_losses[-1] < best
        best = min(best, val_losses[-1])
        checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                          epochs=epoch+1, train_losses=train_losses, val_losses=val_losses,
                          initial_val_loss=initial, config=dict(MODEL_CONFIG), settings=settings,
                          epoch_seconds=epoch_seconds, samples=samples, best_val_loss=best,
                          rng_state=torch.get_rng_state(),
                          cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        if improved:
            save_atomic(checkpoint, output / 'minigpt_best.pth')
        save_atomic(model.state_dict(), output / 'minigpt_weights.pth')
        save_atomic(checkpoint, path)
        print(f'Checkpoint saved: epoch {epoch+1} | {epoch_seconds[-1]/60:.2f} min')
        print('Sample:', repr(sample))
    print(f'Training time: {(time.perf_counter()-start)/60:.2f} min')
    return checkpoint


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/w_pustyni_i_w_puszczy.txt')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--output', default='checkpoints')
    p.add_argument('--device', default=None)
    p.add_argument('--resume', action='store_true')
    train(**vars(p.parse_args()))
