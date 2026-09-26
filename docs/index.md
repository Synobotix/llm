# llm
## Project goal

Large language models are expensive to train and to run. This project explores how far a **small, cheap-to-run model** can go, and documents every step on the way. It has four objectives:

1. **Understand how an LLM is built.** Implement each component by hand (embeddings, causal attention, feed-forward blocks, loss, trainer, text generation) and explain it in the documentation.
2. **Compare two architectures.** Train a *classical* decoder-only transformer and a *BitNet* transformer, whose linear layers use ternary weights (-1, 0, +1), on the same data. The goal is to measure what ternary quantization costs in quality and what it saves in memory.
3. **Make up for limited data and compute with distillation.** Larger "teacher" models generate high-quality training examples from the cleaned text, and the small "student" models learn from them.
4. **Keep everything reproducible.** The same scripts run locally and on Google Colab, with configuration kept in one place and checkpoints that can be resumed.

!!! note "Scope"
    This is a learning and experimentation project, not a production model. Models are deliberately small (a few layers, a few hundred dimensions, short context windows) so that a full training run fits on modest hardware.

## What the project contains

- **Data**: prepare, clean, deduplicate, split and tokenize the C4 dataset.
- **Classical transformer**: a decoder-only (causal) transformer trained with next-token prediction.
- **BitNet transformer**: the same architecture with ternary-weight linear layers (BitLinear), for a much smaller memory footprint.
- **Distillation**: build a training set from teacher-model outputs (served through OpenRouter or Hugging Face), then train the BitNet student on it.

!!! info "Requirements"
    Python 3.10+ and [Poetry](https://python-poetry.org/). See the prerequisites pages below for the full setup.
