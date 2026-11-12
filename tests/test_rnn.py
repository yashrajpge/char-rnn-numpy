"""Sanity tests: run with `python tests/test_rnn.py` from the repo root.

The first run downloads the training text from Project Gutenberg
(needs internet once).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from data import load_text, Vocabulary
from rnn import CharRNN, Adagrad


def test_vocabulary_roundtrip():
    text = load_text()
    vocab = Vocabulary(text)
    sample = text[:1000]
    assert vocab.decode(vocab.encode(sample)) == sample
    assert vocab.size == len(set(text))
    print(f"ok: encode/decode roundtrip over a {vocab.size}-character vocabulary")


def test_analytic_gradient_matches_numeric():
    text = load_text()[:200]
    vocab = Vocabulary(text)
    data = vocab.encode(text)
    model = CharRNN(vocab.size, hidden_size=8, seed=0)
    inputs, targets = data[:5], data[1:6]
    loss0, grads, _ = model.loss(inputs, targets, model.initial_state())
    eps, tol = 1e-5, 1e-4
    rng = np.random.default_rng(0)
    for name in ("Wxh", "Why"):
        flat = model.params[name].ravel()
        for _ in range(8):
            i = rng.integers(0, flat.size)
            old = flat[i]
            flat[i] = old + eps
            plus, _, _ = model.loss(inputs, targets, model.initial_state())
            flat[i] = old - eps
            minus, _, _ = model.loss(inputs, targets, model.initial_state())
            flat[i] = old
            numeric = (plus - minus) / (2 * eps)
            analytic = grads[name].ravel()[i]
            assert abs(numeric - analytic) < tol, (name, i, numeric, analytic)
    print("ok: BPTT gradients match numeric gradients")


def test_loss_decreases():
    text = load_text()[:4000]
    vocab = Vocabulary(text)
    data = vocab.encode(text)
    model = CharRNN(vocab.size, hidden_size=32, seed=1)
    opt = Adagrad(model.params, lr=0.1)
    inputs, targets = data[:25], data[1:26]
    first, _, _ = model.loss(inputs, targets, model.initial_state())
    for _ in range(200):
        _, grads, _ = model.loss(inputs, targets, model.initial_state())
        opt.step(model.params, grads)
    last, _, _ = model.loss(inputs, targets, model.initial_state())
    assert last < first * 0.5, (first, last)
    print(f"ok: loss falls during training ({first:.2f} -> {last:.2f})")


def test_model_is_deterministic():
    text = load_text()[:1000]
    vocab = Vocabulary(text)
    data = vocab.encode(text)
    losses = []
    for _ in range(2):
        model = CharRNN(vocab.size, hidden_size=16, seed=42)
        opt = Adagrad(model.params)
        for _ in range(10):
            loss, grads, _ = model.loss(data[:20], data[1:21], model.initial_state())
            opt.step(model.params, grads)
        losses.append(loss)
    assert losses[0] == losses[1], losses
    print(f"ok: same seed gives identical loss ({losses[0]:.6f})")


def test_sample_shape_and_charset():
    vocab = Vocabulary(load_text())
    model = CharRNN(vocab.size, hidden_size=16, seed=3)
    out = model.sample(0, 200, seed=5, temperature=0.8)
    assert len(out) == 200
    assert all(0 <= i < vocab.size for i in out)
    print("ok: sampling returns 200 in-vocabulary characters")


if __name__ == "__main__":
    test_vocabulary_roundtrip()
    test_analytic_gradient_matches_numeric()
    test_loss_decreases()
    test_model_is_deterministic()
    test_sample_shape_and_charset()
    print("all tests passed")
