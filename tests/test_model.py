import unittest
import torch
from model import MiniGPT
from prepare_data import GPTDataset
from generate import generate_ids


class Tests(unittest.TestCase):
    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in MiniGPT().parameters()), 6537088)

    def test_dataset(self):
        d = GPTDataset(list(range(9)), 4, 4)
        self.assertEqual(len(d), 2)
        self.assertEqual(d[1][0].tolist(), [4, 5, 6, 7])
        self.assertEqual(d[1][1].tolist(), [5, 6, 7, 8])

    def test_forward_and_causality(self):
        m = MiniGPT(vocab_size=20, emb_dim=8, context_length=4, n_layers=2, n_heads=2).eval()
        a = torch.tensor([[1, 2, 3, 4]])
        b = torch.tensor([[1, 2, 5, 6]])
        with torch.no_grad():
            x, y = m(a), m(b)
        self.assertEqual(tuple(x.shape), (1, 4, 20))
        torch.testing.assert_close(x[:, :2], y[:, :2])
        self.assertEqual(tuple(generate_ids(m, a, 3).shape), (1, 7))
        with self.assertRaises(ValueError):
            m(torch.ones(1, 5, dtype=torch.long))

    def test_training_step(self):
        m = MiniGPT(vocab_size=20, emb_dim=8, context_length=4, n_layers=1, n_heads=2)
        opt = torch.optim.AdamW(m.parameters())
        x = torch.tensor([[1, 2, 3, 4]])
        y = torch.tensor([[2, 3, 4, 5]])
        before = m.out_head.weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, 20), y.reshape(-1))
        loss.backward()
        opt.step()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertFalse(torch.equal(before, m.out_head.weight))
