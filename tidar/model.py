import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput
from .masking import build_tidar_hf_mask

class TiDARModel(nn.Module):
    def __init__(self, base_model, tokenizer, diffusion_block_len=4, alpha=1.0):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.convert_tokens_to_ids("<|mask|>")
        if self.mask_token_id == tokenizer.unk_token_id and "<|mask|>" != tokenizer.unk_token:
            raise ValueError("Token <|mask|> not found in tokenizer!")
        self.diffusion_block_len = diffusion_block_len
        self.alpha = alpha

    @property
    def _hf_peft_config_loaded(self) -> bool:
        """Dynamically detect if the underlying base model is configured with PEFT adapters."""
        # Check if the flag is explicitly set on the inner model
        if getattr(self.base_model, "_hf_peft_config_loaded", False):
            return True

        # Check for typical PEFT attributes to handle cases where the flag was not propagated
        if hasattr(self.base_model, "peft_config") or hasattr(self.base_model, "active_adapters"):
            return True

        return False

    def forward(self, input_ids, labels=None, attention_mask=None, output_full_logits=False, **kwargs):
        # Extract input_ids cleanly if passed as kwarg by Trainer frameworks
        if input_ids is None:
            input_ids = kwargs.pop("input_ids", None)
        if input_ids is None:
            raise ValueError("input_ids cannot be None during model forward pass.")

        B, S = input_ids.shape
        device = input_ids.device

        # Append mask tokens for Diffusion
        mask_tokens = torch.full((B, S), self.mask_token_id, dtype=input_ids.dtype, device=device)
        full_input_ids = torch.cat([input_ids, mask_tokens], dim=1)

        # Build TiDAR specific 4D attention mask, incorporating padding
        base_mask = kwargs.pop("attention_mask", attention_mask)
        
        # Build TiDAR specific attention mask
        tidar_mask = build_tidar_hf_mask(B, S, self.diffusion_block_len, device, self.base_model.dtype, base_mask)
        
        # Position IDs repeat for the diffusion part
        position_ids = torch.arange(0, S, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
        full_position_ids = torch.cat([position_ids, position_ids], dim=1)

        kwargs.pop("position_ids", None)
        kwargs.pop("input_ids", None)
        kwargs.pop("labels", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("output_hidden_states", None)

        outputs = self.base_model(
            input_ids=full_input_ids,
            attention_mask=tidar_mask,
            position_ids=full_position_ids,
            output_hidden_states=False,
            use_cache=False, # Disabled during training
            output_router_logits=True, # Ensure MoE models return aux loss
            **kwargs
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # AR Loss (Shifted)
            ar_logits = logits[:, :S-1, :].contiguous()
            ar_labels = labels[:, 1:].contiguous()
            loss_ar = loss_fct(ar_logits.view(-1, ar_logits.size(-1)), ar_labels.view(-1))

            # Diffusion Loss (Unshifted)
            diff_logits = logits[:, S:, :].contiguous()
            diff_labels = labels.contiguous()
            loss_diff = loss_fct(diff_logits.view(-1, diff_logits.size(-1)), diff_labels.view(-1))

            loss = (1.0 / (1.0 + self.alpha)) * (self.alpha * loss_ar + loss_diff)

            # Capture MoE auxiliary loss (load balancing / z-loss) if it exists
            aux_loss = getattr(outputs, "aux_loss", None)
            if aux_loss is not None:
                loss += aux_loss

        ret_logits = logits[:, :S, :].contiguous() if not output_full_logits else logits

        return CausalLMOutput(loss=loss, logits=ret_logits)

    def generate(self, input_ids=None, max_new_tokens=512, temperature=0.5, mode="tidar", **kwargs):
        """Pass through to the TiDAR speculative generation loop, natively supporting HF Trainer."""
        from .generation import TiDARGenerator

        # Instantiate generator dynamically to inherit draft length
        generator = TiDARGenerator(self, draft_len=self.diffusion_block_len)

        # Extract TiDAR specific kwargs
        stop_tokens = kwargs.pop("stop_tokens", None)
        regex_pattern = kwargs.pop("regex_pattern", None)
        log_diffusion = kwargs.pop("log_diffusion", False) # False by default to prevent spam during HF Eval
        trust_ar_ratio = kwargs.pop("trust_ar_ratio", None)

        # Clean up HF specific evaluation kwargs that we manage manually
        kwargs.pop("attention_mask", None)
        kwargs.pop("labels", None)
        kwargs.pop("use_cache", None)

        if input_ids is None:
            if "prompt" in kwargs:
                input_ids = self.tokenizer(kwargs.pop("prompt"), return_tensors="pt").input_ids.to(self.base_model.device)
            else:
                raise ValueError("Either input_ids or prompt must be provided to generate.")

        # Execute generation
        output_ids = generator.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_tokens=stop_tokens,
            regex_pattern=regex_pattern,
            log_diffusion=log_diffusion,
            trust_ar_ratio=trust_ar_ratio,
            mode=mode,
            return_tensors=True # Forces the generator to return HF compatible tensors
        )

        return output_ids

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If the base model hasn't been set yet (e.g., during initialization), raise standard error
            if name == "base_model":
                raise
            # Forward the call/attribute to the Hugging Face base model
            if hasattr(self.base_model, name):
                return getattr(self.base_model, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
