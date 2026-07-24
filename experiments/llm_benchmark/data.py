from pathlib import Path

import torch

from experiments.llm_benchmark.config import DataConfig


BUILTIN_TINY_TEXT = """
In a quiet laboratory, a small model learned to predict the next token.
The experiment was repeated with different random seeds so the results would
not depend on a lucky initialization. Curvature can help when the landscape is
ill conditioned, but every optimizer must pay for the computation it performs.

The researcher measured validation loss, gradient norm, memory, throughput, and
wall clock time. A fair benchmark reports both convergence and cost. It also
shows failure cases, because scientific claims should survive difficult tests.

Tiny stories are useful for debugging language models. A robot sorted colored
blocks, then wrote careful notes about each step. The notes were simple, but
the repeated structure let the model learn patterns beyond random guessing.
"""


def _encode_bytes(text: str, vocab_size: int) -> torch.Tensor:
    values = [byte % vocab_size for byte in text.encode("utf-8", errors="ignore")]
    return torch.tensor(values, dtype=torch.long)


def _load_local_text_dataset(name: str, data_dir: Path, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    candidates = [
        data_dir / name / "train.txt",
        data_dir / f"{name}_train.txt",
        data_dir / "train.txt",
    ]
    train_path = next((path for path in candidates if path.exists()), None)
    val_candidates = [
        data_dir / name / "valid.txt",
        data_dir / name / "val.txt",
        data_dir / f"{name}_valid.txt",
        data_dir / f"{name}_val.txt",
        data_dir / "valid.txt",
        data_dir / "val.txt",
    ]
    val_path = next((path for path in val_candidates if path.exists()), None)
    if train_path is None:
        return None
    train = _encode_bytes(train_path.read_text(encoding="utf-8", errors="ignore"), vocab_size)
    if val_path is not None:
        val = _encode_bytes(val_path.read_text(encoding="utf-8", errors="ignore"), vocab_size)
    else:
        split = max(int(0.9 * len(train)), 1)
        train, val = train[:split], train[split:]
    if len(train) < 2 or len(val) < 2:
        raise ValueError(f"Dataset at {train_path} is too small for language modeling")
    return train, val


def _builtin_text_tokens(seed: int, train_tokens: int, val_tokens: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    needed = train_tokens + val_tokens
    repeats = max(needed // len(BUILTIN_TINY_TEXT) + 2, 4)
    tokens = _encode_bytes((BUILTIN_TINY_TEXT.strip() + "\n") * repeats, vocab_size)
    generator = torch.Generator().manual_seed(seed)
    offset = int(torch.randint(0, max(len(tokens) - 1, 1), (1,), generator=generator))
    tokens = torch.cat([tokens[offset:], tokens[:offset]])
    return tokens[:train_tokens].clone(), tokens[train_tokens:train_tokens + val_tokens].clone()


def _synthetic_tokens(seed: int, length: int, vocab_size: int, noise: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    motif_count = 32
    motif_length = 24
    motifs = torch.randint(0, vocab_size, (motif_count, motif_length), generator=generator)
    choices = torch.randint(0, motif_count, (length // motif_length + 1,), generator=generator)
    stream = motifs[choices].flatten()[:length].clone()
    noise_mask = torch.rand(length, generator=generator) < noise
    stream[noise_mask] = torch.randint(0, vocab_size, (int(noise_mask.sum()),), generator=generator)
    return stream.long()


def load_token_data(config: DataConfig, vocab_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    data_dir = Path(config.data_dir)
    if config.dataset.lower() in {"wikitext2", "wikitext-2", "tinystories"}:
        loaded = _load_local_text_dataset(config.dataset.lower(), data_dir, vocab_size)
        if loaded is not None:
            train, val = loaded
            return train, val, config.dataset.lower()
    if config.dataset.lower() in {"tiny_text", "builtin_text", "text"}:
        train, val = _builtin_text_tokens(seed, config.train_tokens, config.val_tokens, vocab_size)
        return train, val, "builtin_text"
    train = _synthetic_tokens(seed, config.train_tokens, vocab_size, config.synthetic_noise)
    val = _synthetic_tokens(seed + 10_000, config.val_tokens, vocab_size, config.synthetic_noise)
    return train, val, "synthetic"


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("Token data is too short for configured block_size")
    max_start = len(data) - block_size - 1
    start = (step * batch_size * block_size) % max_start
    offsets = (start + torch.arange(batch_size) * block_size) % max_start
    x = torch.stack([data[int(i) : int(i) + block_size] for i in offsets])
    y = torch.stack([data[int(i) + 1 : int(i) + block_size + 1] for i in offsets])
    return x.to(device), y.to(device)
