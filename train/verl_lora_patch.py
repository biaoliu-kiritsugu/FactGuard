"""FactGuard compatibility patches for verl's current FSDP integration.

verl 0.9.0.dev exposes rank, alpha, and target modules but does not pass a
dropout value to PEFT's ``LoraConfig``.  The SFT launcher imports this module
through ``model.external_lib`` and supplies FACTGUARD_LORA_DROPOUT=0.05.

InternLM's tokenizer also contains chat-template tokens beyond the checkpoint's
embedding size. LLaMA-Factory resizes those embeddings automatically; verl's
current FSDP loader does not. Resize before FSDP wraps the model when needed.
"""

from __future__ import annotations

import os

from transformers import PreTrainedModel
from verl.workers.engine.fsdp import transformer_impl


# Transformers 5 expects a property introduced after GLM-4-9B's bundled
# remote-code class was published. The model declares no tied-weight mapping,
# so an empty mapping preserves the old behavior during checkpoint loading.
if not getattr(PreTrainedModel.mark_tied_weights_as_initialized, "_factguard_remote_code_patch", False):
    _original_mark_tied_weights = PreTrainedModel.mark_tied_weights_as_initialized

    def _mark_tied_weights_compat(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _original_mark_tied_weights(self, loading_info)

    _mark_tied_weights_compat._factguard_remote_code_patch = True
    PreTrainedModel.mark_tied_weights_as_initialized = _mark_tied_weights_compat


if not getattr(transformer_impl.FSDPEngine._build_module, "_factguard_vocab_patch", False):
    _original_build_module = transformer_impl.FSDPEngine._build_module

    def _build_module_with_tokenizer_vocab(self):
        module = _original_build_module(self)
        tokenizer_size = len(self.model_config.tokenizer)
        embedding_size = module.get_input_embeddings().num_embeddings
        if tokenizer_size > embedding_size:
            module.resize_token_embeddings(tokenizer_size, mean_resizing=False)
            print(
                "FactGuard verl extension: resized token embeddings "
                f"from {embedding_size} to {tokenizer_size}"
            )
        return module

    _build_module_with_tokenizer_vocab._factguard_vocab_patch = True
    transformer_impl.FSDPEngine._build_module = _build_module_with_tokenizer_vocab


if not getattr(transformer_impl.LoraConfig, "_factguard_dropout_patch", False):
    _original_lora_config = transformer_impl.LoraConfig

    def _lora_config_with_dropout(*args, **kwargs):
        dropout = float(os.environ.get("FACTGUARD_LORA_DROPOUT", "0.05"))
        kwargs.setdefault("lora_dropout", dropout)
        return _original_lora_config(*args, **kwargs)

    _lora_config_with_dropout._factguard_dropout_patch = True
    transformer_impl.LoraConfig = _lora_config_with_dropout
    print(
        "FactGuard verl extension: PEFT LoraConfig uses "
        f"lora_dropout={os.environ.get('FACTGUARD_LORA_DROPOUT', '0.05')}"
    )
