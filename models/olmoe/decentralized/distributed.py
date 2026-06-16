from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DecodeState:
    past_key_values: object | None
    attention_mask: torch.Tensor

class DistributedOlmoe:
    """
    High-level inference engine for decentralized OLMoE.

    Responsibilities:
    - run one distributed forward pass
    - maintain KV-cache during generation
    - prefill prompt once
    - decode one token at a time
    - provide generate() API for GSM8K / normal generation
    - provide scoring utilities for MMLU / log-prob evaluation
    """
    def __init__(
        self,
        attg_module,
        router,
        tokenizer=None,
        device: Optional[torch.device] = None,
    ):
        self.attg = attg_module.eval()
        self.router = router
        self.tokenizer = tokenizer
        self.config = attg_module.config

        if device is None:
            # fallback: infer from model parameters
            device = next(attg_module.parameters()).device

        self.device = device

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
    ):
        """
        Distributed forward.

        Prefill:
            input_ids shape = [batch, prompt_len]

        Decode:
            input_ids shape = [batch, 1]
        """
        input_ids = input_ids.to(self.device)

        if attention_mask is None:
            past_len = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            attention_mask = torch.ones(
                input_ids.shape[0],
                past_len + input_ids.shape[1],
                device=self.device,
                dtype=torch.long,
            )
        else:
            attention_mask = attention_mask.to(self.device)

        preprocessed = self.attg.preprocess_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        hidden_states = preprocessed["inputs_embeds"]

        for layer_idx in range(self.config.num_hidden_layers):
            ffn_residual, req = self.attg.forward(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                causal_mask=preprocessed["causal_mask"],
                position_ids=preprocessed["position_ids"],
                position_embeddings=preprocessed["position_embeddings"],
                past_key_values=preprocessed["past_key_values"],
            )

            expert_output = self.router.forward(
                layer_idx=req["layer_idx"],
                hidden_states=req["hidden_states"],
                top_k_index=req["top_k_index"],
                top_k_weights=req["top_k_weights"],
            )

            hidden_states = self.attg.merge_expert_output(
                ffn_residual=ffn_residual,
                expert_output=expert_output,
                original_shape=req["original_shape"],
            )

        logits = self.attg.final_logits(hidden_states)

        return logits, DecodeState(
            past_key_values=preprocessed["past_key_values"],
            attention_mask=attention_mask,
        )

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor) -> tuple[torch.Tensor, DecodeState]:
        attention_mask = torch.ones_like(input_ids, device=self.device)
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )

    @torch.inference_mode()
    def decode_step(
        self,
        next_token: torch.LongTensor,
        state: DecodeState,
    ) -> tuple[torch.Tensor, DecodeState]:
        """
        Decode exactly one token using existing KV-cache.
        """
        next_token = next_token.to(self.device)

        attention_mask = torch.cat(
            [
                state.attention_mask,
                torch.ones(
                    state.attention_mask.shape[0],
                    1,
                    device=self.device,
                    dtype=state.attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        return self.forward(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            use_cache=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        logits, state = self.prefill(input_ids)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_token]

        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            logits, state = self.decode_step(next_token, state)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)

        return torch.cat([input_ids.to(self.device)] + generated, dim=1)

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        if self.tokenizer is None:
            raise ValueError("generate_text requires tokenizer")

        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)

        output_ids = self.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)