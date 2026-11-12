"""A character-level recurrent neural network written from scratch.

One tanh hidden layer, softmax output over the character vocabulary.
Forward pass, backpropagation through time and the Adagrad update are
implemented by hand - no deep-learning framework anywhere.
"""

import numpy as np


class CharRNN:
    def __init__(self, vocab_size, hidden_size=128, seed=42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.params = {
            "Wxh": rng.standard_normal((hidden_size, vocab_size)) * 0.01,
            "Whh": rng.standard_normal((hidden_size, hidden_size)) * 0.01,
            "Why": rng.standard_normal((vocab_size, hidden_size)) * 0.01,
            "bh": np.zeros(hidden_size),
            "by": np.zeros(vocab_size),
        }

    def initial_state(self):
        return np.zeros(self.hidden_size)

    def loss(self, inputs, targets, hprev):
        """Cross-entropy over one sequence, with BPTT gradients.

        inputs/targets are integer lists of the same length; targets are
        the inputs shifted one character into the future.
        Returns (loss, grads, last_hidden).
        """
        xs, hs, ps = {}, {}, {}
        hs[-1] = hprev.copy()
        loss = 0.0
        p = self.params
        for t in range(len(inputs)):
            x = np.zeros(self.vocab_size)
            x[inputs[t]] = 1.0
            xs[t] = x
            hs[t] = np.tanh(p["Wxh"] @ x + p["Whh"] @ hs[t - 1] + p["bh"])
            logits = p["Why"] @ hs[t] + p["by"]
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            ps[t] = probs
            loss += -np.log(probs[targets[t]] + 1e-12)

        grads = {k: np.zeros_like(v) for k, v in p.items()}
        dhnext = np.zeros(self.hidden_size)
        for t in reversed(range(len(inputs))):
            dlogits = ps[t].copy()
            dlogits[targets[t]] -= 1.0
            grads["Why"] += np.outer(dlogits, hs[t])
            grads["by"] += dlogits
            dh = p["Why"].T @ dlogits + dhnext
            dhraw = (1.0 - hs[t] ** 2) * dh
            grads["bh"] += dhraw
            grads["Wxh"] += np.outer(dhraw, xs[t])
            grads["Whh"] += np.outer(dhraw, hs[t - 1])
            dhnext = p["Whh"].T @ dhraw
        for g in grads.values():
            np.clip(g, -5.0, 5.0, out=g)
        return float(loss), grads, hs[len(inputs) - 1]

    def sample(self, start_index, n, seed, temperature=1.0, h=None):
        """Generate n characters starting from one seed character index."""
        rng = np.random.default_rng(seed)
        p = self.params
        h = self.initial_state() if h is None else h
        idx = start_index
        out = []
        for _ in range(n):
            x = np.zeros(self.vocab_size)
            x[idx] = 1.0
            h = np.tanh(p["Wxh"] @ x + p["Whh"] @ h + p["bh"])
            logits = (p["Why"] @ h + p["by"]) / temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            idx = rng.choice(self.vocab_size, p=probs)
            out.append(int(idx))
        return out


class Adagrad:
    """Adagrad updater: per-parameter adaptive learning rates."""

    def __init__(self, params, lr=0.1, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.memory = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        for k in params:
            self.memory[k] += grads[k] * grads[k]
            params[k] -= self.lr * grads[k] / np.sqrt(self.memory[k] + self.eps)
