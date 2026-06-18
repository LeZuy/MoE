from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GenerationMixin, GenerationConfig, PreTrainedModel

@dataclass
class DecodeState:
    past_key_values: object | None
    attention_mask: torch.Tensor

class DistributedOlmoe(PreTrainedModel, GenerationMixin):
    main_input_name = "input_ids"
    _supports_sdpa = True 
    _supports_flash_attn = False
    _supports_grouped_mm = False
    def __init__(self, attg_module, router, tokenizer=None, device=None):
        super().__init__(attg_module.config) 
        self.attg = attg_module.eval()
        self.router = router
        self.tokenizer = tokenizer
        self.config = attg_module.config
        self.generation_config = GenerationConfig.from_model_config(self.config)
        # self.device = device or next(attg_module.parameters()).device
        
    def _check_and_adjust_attn_implementation(self, *args, **kwargs):
        return "eager"

    def _check_and_adjust_experts_implementation(self, *args, **kwargs):
        return "eager"
    
    def can_generate(self) -> bool:
        return True
    
    def _init_weights(self, module):
        pass # Weight loaded at AttnGate Module 

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Distributed forward.

        Prefill:
            input_ids shape = [batch, prompt_len]

        Decode:
            input_ids shape = [batch, 1]
        """
        input_ids = input_ids.to(self.device)

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

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=preprocessed["past_key_values"],
        )

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