# Char-RNN Language Model (NumPy from scratch)

A character-level language model: a recurrent neural network with one
tanh hidden layer, trained with backpropagation through time written by
hand - no TensorFlow, no PyTorch, no framework at all. The same math
that powers modern language models, small enough to read end to end.

Built by Yashraj Pangave.

## What it does

- Trains a vanilla RNN (128 hidden units, 50-character windows, Adagrad
  updates) on Lewis Carroll's *Alice's Adventures in Wonderland*:
  144,599 characters of book text with a 75-character vocabulary, split
  into a 90% training prefix and a 10% held-out tail.
- With the default settings (`python src/train.py`, 20,000 steps, seed
  42) the measured result is:

  | split    | loss (mean cross-entropy per character) |
  |----------|------------------------------------------|
  | train    | 2.1594                                   |
  | held-out | 2.1796                                   |

  The untrained baseline is ln(75) = 4.3175, so the model has learned
  real structure. The run is deterministic: the same command reproduces
  the same numbers in about 90 seconds on a laptop CPU.
- `python src/generate.py` samples new text character by character from
  the trained weights. Exact output of `python src/generate.py` (400
  characters, temperature 0.8, seed 7), unedited:

  ```
  souida sine celiyd waid Aling of
  as has crhiot mise ame thourned be was then’t aing the gabserat hadteed?” sarrtat
  siigl
  shere hais, at wastinghes ipy the ward atigheme wase and,” the taace an: So hiote! mastoaid theng to mle shisdatlabneet cer hartapime has, So bacr criiglek the gasf aroud the—” the sire ideed shes.
  Budent ay aphe the haghek you had maing seilk.

  “You wory iou did nousother as th
  ```

  The model has learned English-looking syllables, quotation marks and
  paragraph breaks, but not meaning - that is exactly what a 128-unit
  character RNN can do at this scale, and seeing that boundary is the
  point of the project.

## Quick start

```bash
pip install -r requirements.txt   # numpy only
python src/train.py               # downloads the text once, trains ~90s, saves weights
python src/generate.py            # prints 400 fresh characters
python src/generate.py --chars 1000 --temperature 0.5   # safer, more repetitive
python src/generate.py --temperature 1.2                 # wilder
python tests/test_rnn.py          # sanity tests incl. a numeric gradient check
```

The training text is not bundled with this repository. The first run
downloads it (about 170 KB) directly from its owning source, Project
Gutenberg, and caches it in `data/`, which is git-ignored. Internet
access is needed for that first download only.

## Data source and terms

*Alice's Adventures in Wonderland* - Lewis Carroll (1865). The text is
in the public domain; it is downloaded at runtime from Project Gutenberg
eBook #11: https://www.gutenberg.org/cache/epub/11/pg11.txt

This project does not redistribute the text. The Project Gutenberg
header and footer boilerplate is stripped before training (see
`src/data.py`), so the model learns the book itself. The MIT license in
this repository covers the code only, not the text; if you reuse the
downloaded file, respect the terms on the Project Gutenberg site.

## Project layout

```
src/rnn.py       the model: forward pass, BPTT, gradient clipping,
                 sampling, and the Adagrad updater
src/data.py      download-on-first-use loader, boilerplate stripping,
                 character vocabulary
src/train.py     training loop with exact train/held-out loss reporting
src/generate.py  sample new text from saved weights
tests/           sanity tests: vocabulary roundtrip, BPTT gradients vs
                 numeric gradients, learning, determinism, sampling
data/            local text cache (git-ignored, created on first run)
models/          trained weights (git-ignored, created by src/train.py)
```

## Why from scratch?

Backpropagation through time is where most people's understanding of
recurrent networks goes fuzzy. Writing the forward caches, the
reversed-time gradient loop and the clipping by hand - and then proving
them against numeric gradients in the tests - turns that fog into code
you can trust.
