"""FactGuard-specific, answer-masked SFT dataset for verl.

verl's stock ``MultiTurnSFTDataset`` currently right-truncates no-padding
samples even when left truncation is requested.  FactGuard puts the question
at the end of a potentially long document, so right truncation can remove both
the question and the complete assistant target.  This dataset keeps the tail
of over-length examples and masks every token before the assistant answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset


def _as_token_ids(value: Any) -> list[int]:
    # transformers 5 returns BatchEncoding (a Mapping, not a plain dict)
    # from some fast-tokenizer chat templates.
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.squeeze(0).tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


class FactGuardSFTDataset(Dataset):
    """Two-turn message dataset with assistant-only loss masking."""

    def __init__(
        self,
        parquet_files,
        tokenizer,
        config: dict | None = None,
        processor=None,
        max_samples: int = -1,
    ) -> None:
        del processor
        self.config = config or {}
        self.tokenizer = tokenizer
        self.max_length = int(self.config.get("max_length", 32768))
        self.messages_key = self.config.get("messages_key", "messages")
        self.preserve_bos = bool(self.config.get("preserve_bos", True))
        self.pad_mode = self.config.get("pad_mode", "no_padding")

        if self.pad_mode not in {"no_padding", "right"}:
            raise ValueError("FactGuardSFTDataset supports data.pad_mode=no_padding or right")
        if self.config.get("truncation", "left") != "left":
            raise ValueError("FactGuardSFTDataset requires data.truncation=left")

        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(Path(path), columns=[self.messages_key]) for path in parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        if 0 < max_samples < len(self.dataframe):
            self.dataframe = self.dataframe.iloc[:max_samples].reset_index(drop=True)
        print(f"FactGuardSFTDataset: {len(self.dataframe)} rows, max_length={self.max_length}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _tokenize(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError("Each row must end with an assistant message")

        prompt_ids = _as_token_ids(
            self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        full_ids = _as_token_ids(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )

        # Normally prompt_ids is an exact prefix of full_ids.  The longest
        # common prefix fallback also handles templates that alter a separator
        # when the assistant content is present.
        prefix_len = 0
        for prompt_token, full_token in zip(prompt_ids, full_ids):
            if prompt_token != full_token:
                break
            prefix_len += 1
        if prefix_len == 0 or prefix_len >= len(full_ids):
            raise ValueError(
                "Could not identify the assistant span from this tokenizer's chat template: "
                f"common_prefix={prefix_len}, full_length={len(full_ids)}"
            )
        loss_mask = [0] * prefix_len + [1] * (len(full_ids) - prefix_len)
        return full_ids, loss_mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        messages = self.dataframe.iloc[index][self.messages_key]
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        messages = [dict(message) for message in messages]
        input_ids, loss_mask = self._tokenize(messages)

        if len(input_ids) > self.max_length:
            if self.preserve_bos and self.max_length > 1:
                input_ids = [input_ids[0], *input_ids[-(self.max_length - 1) :]]
                loss_mask = [0, *loss_mask[-(self.max_length - 1) :]]
            else:
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]

        if not any(loss_mask):
            raise ValueError(f"Sample {index} has no assistant tokens after truncation")

        sequence_length = len(input_ids)
        position_ids = list(range(sequence_length))
        attention_mask = [1] * sequence_length
        if self.pad_mode == "right" and sequence_length < self.max_length:
            padding_length = self.max_length - sequence_length
            pad_token_id = self.tokenizer.pad_token_id or 0
            input_ids.extend([pad_token_id] * padding_length)
            loss_mask.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
            position_ids.extend([0] * padding_length)

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }
        if self.pad_mode == "right":
            result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        return result
