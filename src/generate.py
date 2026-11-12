"""Generate text from the trained character RNN.

Usage: python src/generate.py [--chars 400] [--temperature 0.8] [--seed 7]
Run src/train.py first so models/char_rnn.npz exists.
"""

import argparse
import os

import numpy as np

from rnn import CharRNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chars", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prime", default="\n", help="one seed character")
    args = parser.parse_args()

    if not os.path.exists("models/char_rnn.npz"):
        print("no trained weights found - run python src/train.py first")
        raise SystemExit(1)
    blob = np.load("models/char_rnn.npz")
    chars = [chr(int(c)) for c in blob["chars"]]
    model = CharRNN(len(chars), hidden_size=int(blob["hidden"][0]))
    for k in model.params:
        model.params[k] = blob[k]

    if len(args.prime) != 1 or args.prime not in chars:
        print(f"--prime must be exactly one character from the vocabulary")
        raise SystemExit(1)
    start = chars.index(args.prime)
    out = model.sample(start, args.chars, seed=args.seed, temperature=args.temperature)
    print("".join(chars[i] for i in out))


if __name__ == "__main__":
    main()
