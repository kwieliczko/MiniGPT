# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path
import time
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


def train(data, epochs=5, batch_size=4, lr=0.0003, output='checkpoints', device=None):
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError('Epochs, batch size and learning rate must be positive')
    torch.manual_seed(SEED)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    train_loader, val_loader = make_loaders(data, MODEL_CONFIG['context_length'], batch_size)
    model = MiniGPT(**MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print('Device:', device, '| Parameters:', sum(p.numel() for p in model.parameters()))
    initial = evaluate_model(model, val_loader, device)
    print(f'Initial validation loss: {initial:.4f}')
    train_losses, val_losses = [], []
    start = time.perf_counter()
    for epoch in range(epochs):
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
        print(f'Epoch {epoch+1}/{epochs} | Train loss: {train_losses[-1]:.4f} | Validation loss: {val_losses[-1]:.4f}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / 'minigpt_weights.pth')
    checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                      epochs=epochs, train_losses=train_losses, val_losses=val_losses,
                      initial_val_loss=initial, config=dict(MODEL_CONFIG))
    torch.save(checkpoint, output / 'minigpt_checkpoint.pth')
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
    train(**vars(p.parse_args()))
