"""Train the character RNN on the book text.

Usage: python src/train.py [--steps 20000] [--hidden 128] [--seq-len 50]
Splits the text into a 90% training prefix and a 10% held-out tail,
trains with Adagrad on random windows, then reports the exact mean loss
on fixed evaluation windows from both parts. Deterministic: the same
command reproduces the same numbers.
"""

import argparse
import os

import numpy as np

from data import load_text, Vocabulary
from rnn import CharRNN, Adagrad


def mean_loss(model, data, seq_len, windows, rng):
    """Mean cross-entropy per character over fixed, non-overlapping windows."""
    starts = rng.integers(0, len(data) - seq_len - 1, size=windows)
    total = 0.0
    for s in starts:
        loss, _, _ = model.loss(
            data[s : s + seq_len], data[s + 1 : s + seq_len + 1], model.initial_state()
        )
        total += loss / seq_len
    return total / windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    text = load_text()
    vocab = Vocabulary(text)
    data = vocab.encode(text)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    print(f"text: {len(data):,} characters, vocabulary: {vocab.size}")
    print(f"train {len(train_data):,} / held-out {len(val_data):,}")

    model = CharRNN(vocab.size, hidden_size=args.hidden, seed=args.seed)
    opt = Adagrad(model.params, lr=args.lr)
    rng = np.random.default_rng(args.seed)

    smooth = None
    for step in range(1, args.steps + 1):
        s = rng.integers(0, len(train_data) - args.seq_len - 1)
        loss, grads, _ = model.loss(
            train_data[s : s + args.seq_len],
            train_data[s + 1 : s + args.seq_len + 1],
            model.initial_state(),
        )
        per_char = loss / args.seq_len
        smooth = per_char if smooth is None else 0.999 * smooth + 0.001 * per_char
        opt.step(model.params, grads)
        if step == 1 or step % 1000 == 0:
            print(f"step {step:>5}: smoothed train loss {smooth:.4f}")

    eval_rng = np.random.default_rng(args.seed + 1)
    train_loss = mean_loss(model, train_data, args.seq_len, 50, eval_rng)
    val_loss = mean_loss(model, val_data, args.seq_len, 50, eval_rng)
    print(f"final: train loss {train_loss:.4f}, held-out loss {val_loss:.4f}")
    print(f"initial-loss baseline ln(vocab) = {np.log(vocab.size):.4f}")

    os.makedirs("models", exist_ok=True)
    np.savez(
        "models/char_rnn.npz",
        chars=np.array([ord(c) for c in vocab.chars]),
        hidden=np.array([args.hidden]),
        **model.params,
    )
    print("saved weights to models/char_rnn.npz")


if __name__ == "__main__":
    main()
