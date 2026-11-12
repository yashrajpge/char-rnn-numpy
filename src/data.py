"""Text loader: downloads a public-domain book from Project Gutenberg.

The text is not bundled with this repository. The first run downloads it
(about 170 KB) directly from its owning source and caches it in data/,
which is git-ignored. The Project Gutenberg header and footer
boilerplate are stripped so the model trains on the book text itself.
Internet access is needed for that first download only.
"""

import os
import urllib.request

import numpy as np

URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
START_MARK = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARK = "*** END OF THE PROJECT GUTENBERG EBOOK"


def _ensure_cached(cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "alice.txt")
    if not os.path.exists(path):
        print("downloading the text from Project Gutenberg (one time only)...")
        urllib.request.urlretrieve(URL, path)
    return path


def load_text(cache_dir="data"):
    """Return the book text as one Python string, boilerplate stripped."""
    raw = open(_ensure_cached(cache_dir), encoding="utf-8").read()
    start = raw.find(START_MARK)
    end = raw.find(END_MARK)
    if start != -1 and end != -1:
        raw = raw[raw.find("\n", start) + 1 : end]
    return raw.strip()


class Vocabulary:
    """Character <-> index mapping for a text."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, indices):
        return "".join(self.chars[int(i)] for i in indices)
