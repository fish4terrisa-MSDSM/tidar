import torch
import torch.nn.functional as F
from transformers import DynamicCache
import time

class TiDARGenerator:
    def __init__(self, tidar_model, draft_len=4):
        self.model = tidar_model.base_model # Extract base Model for manual KV cache control
        self.tokenizer = tidar_model.tokenizer
        self.mask_token_id = tidar_model.mask_token_id
        self.draft_len = draft_len
        self.device = self.model.device

    def generate(self, prompt=None, input_ids=None, max_new_tokens=512, temperature=0.0,
                 stop_tokens=None, regex_pattern=None, log_diffusion=True, trust_ar_ratio=None, mode="tidar", return_tensors=False):
        """
        Dispatcher for Generation.
        mode can be: "tidar", "ar", or "diffusion" (block diffusion).
        """
        if input_ids is None:
            if prompt is None:
                raise ValueError("Either prompt or input_ids must be provided.")
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_token_id]

        if mode == "tidar":
            generated_ids = self._generate_tidar(input_ids, max_new_tokens, temperature, stop_tokens, regex_pattern, log_diffusion, trust_ar_ratio)
        elif mode == "ar":
            generated_ids = self._generate_ar(input_ids, max_new_tokens, temperature, stop_tokens, regex_pattern)
        elif mode == "diffusion":
            generated_ids = self._generate_diffusion(input_ids, max_new_tokens, temperature, stop_tokens, regex_pattern, log_diffusion)
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

        if return_tensors:
            return generated_ids
        else:
            seq_len = input_ids.shape[1]
            return self.tokenizer.decode(generated_ids[0, seq_len:], skip_special_tokens=False)

    @torch.no_grad()
    def _generate_ar(self, input_ids, max_new_tokens, temperature, stop_tokens, regex_pattern):
        """Pure Autoregressive Generation."""
        B, seq_len = input_ids.shape
        generated_ids = input_ids.clone()
        past_key_values = DynamicCache()

        logits_processor = None
        if regex_pattern:
            from .utils import RegexLogitsProcessor
            logits_processor = RegexLogitsProcessor(regex_pattern, self.tokenizer, stop_tokens, prompt_len=seq_len)

        last_token = input_ids
        for _ in range(max_new_tokens):
            outputs = self.model(last_token, use_cache=True, past_key_values=past_key_values)
            next_logits = outputs.logits[:, -1:, :]

            if logits_processor:
                next_logits = logits_processor(generated_ids, next_logits, current_draft=None)

            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                last_token = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
            else:
                last_token = next_logits.argmax(dim=-1).view(B, 1)

            generated_ids = torch.cat([generated_ids, last_token], dim=1)

            if last_token[0].item() in stop_tokens:
                break

        return generated_ids

    @torch.no_grad()
    def _generate_diffusion(self, input_ids, max_new_tokens, temperature, stop_tokens, regex_pattern, log_diffusion):
        """Pure Block Diffusion Generation"""
        B, seq_len = input_ids.shape
        generated_ids = input_ids.clone()
        k = self.draft_len
        dtype = self.model.dtype

        steps = 0
        start_time = time.time()

        while generated_ids.shape[1] < seq_len + max_new_tokens:
            steps += 1
            masks = torch.full((B, k), self.mask_token_id, device=self.device)
            model_input = torch.cat([generated_ids, masks], dim=1)
            S = model_input.shape[1]

            # Causal Prefix + Bidirectional Block Masking
            attn_mask = torch.full((B, 1, S, S), torch.finfo(dtype).min, device=self.device, dtype=dtype)
            causal = torch.tril(torch.ones((S-k, S-k), device=self.device))
            attn_mask[:, :, :S-k, :S-k] = attn_mask[:, :, :S-k, :S-k].masked_fill(causal == 1, 0.0)
            attn_mask[:, :, S-k:, :S-k] = 0.0 # Masks can see the prefix
            attn_mask[:, :, S-k:, S-k:] = 0.0 # Masks can see each other bidirectionally

            # Forward without KV cache (safest approach since bidirectional masks invalidate causal cache)
            outputs = self.model(model_input, attention_mask=attn_mask, use_cache=False)
            diff_logits = outputs.logits[:, -k:, :]

            if temperature > 0:
                probs = F.softmax(diff_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, k)
            else:
                next_tokens = diff_logits.argmax(dim=-1)

            if log_diffusion:
                accept_str = self.tokenizer.decode(next_tokens[0].tolist(), skip_special_tokens=True)
                print(f"[Block Diffusion Step {steps}] Drafted & Accepted Block: '{accept_str}'")

            # Check stop conditions
            added_tokens = next_tokens[0].tolist()
            stopped = False
            for i, tok in enumerate(added_tokens):
                if tok in stop_tokens:
                    generated_ids = torch.cat([generated_ids, torch.tensor([added_tokens[:i+1]], device=self.device)], dim=1)
                    stopped = True
                    break

            if not stopped:
                generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            else:
                break

        if log_diffusion:
            duration = time.time() - start_time
            gen_tokens = generated_ids.shape[1] - seq_len
            print(f"\n[Diffusion Stats] Generated {gen_tokens} tokens in {duration:.2f}s ({gen_tokens/duration:.2f} t/s).")

        return generated_ids

    @torch.no_grad()
    def _generate_tidar(self, input_ids, max_new_tokens=512, temperature=0.0, stop_tokens=None, regex_pattern=None, log_diffusion=True, trust_ar_ratio=None):
        """
        Full Speculative TiDAR Generation Loop.
        temperature=0.0 uses greedy sampling. >0.0 uses multinomial sampling.
        """
        B, seq_len = input_ids.shape
        if B > 1:
            raise ValueError("TiDAR speculative decoding currently optimized for Batch Size = 1.")

        generated_ids = input_ids.clone()
        past_key_values = DynamicCache()

        logits_processor = None
        if regex_pattern:
            from .utils import RegexLogitsProcessor
            logits_processor = RegexLogitsProcessor(regex_pattern, self.tokenizer, stop_tokens, prompt_len=seq_len)

        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_token_id]

        # Prefill
        outputs = self.model(input_ids, use_cache=True, past_key_values=past_key_values)
        
        # Predict the very first token
        first_ar_logits = outputs.logits[:, -1, :]

        # Ensure Regex validation is applied to the very first prefill token
        if logits_processor:
            first_ar_logits_seq = first_ar_logits.unsqueeze(1) # shape [B, 1, V]
            first_ar_logits_seq = logits_processor(generated_ids, first_ar_logits_seq, current_draft=None)
            first_ar_logits = first_ar_logits_seq.squeeze(1)

        if temperature > 0:
            probs = F.softmax(first_ar_logits / temperature, dim=-1)
            last_token = torch.multinomial(probs, num_samples=1)
        else:
            last_token = first_ar_logits.argmax(dim=-1, keepdim=True)
            
        generated_ids = torch.cat([generated_ids, last_token], dim=1)
        
        # Initialize a dummy first draft (the AR model will correct it if it's wrong)
        k = self.draft_len
        current_draft = torch.full((B, k), self.mask_token_id, device=self.device)
        current_draft_probs = None
        current_draft_probs_full = None

        steps = 0
        total_accepted_drafts = 0
        start_time = time.time()

        # Full Speculative Loop
        while generated_ids.shape[1] < seq_len + max_new_tokens:
            steps += 1
            
            # Build Model Input: [Last_Token (1)] + [Drafts (k)] + [Masks ((k+1)*(k+1))]
            # We draft for all possible matching outcomes (0 matches, 1 match, ..., k-1 matches)
            # Predict (k+1) masks to correctly discard the token matching the new AR target and retain exactly 'k' shifted drafts.
            masks = torch.full((B, (k + 1) * (k + 1)), self.mask_token_id, device=self.device)
            model_input = torch.cat([last_token, current_draft, masks], dim=1)
            
            Q = 1 + k + (k + 1) * (k + 1)
            L = past_key_values.get_seq_length()
            
            # Build Position IDs
            pos_ids_list = [L] # Last token
            pos_ids_list.extend([L + 1 + j for j in range(k)]) # Drafts
            for i in range(k + 1):
                # Mask Block i (Conditioned on i accepted drafts)
                pos_ids_list.extend([L + i + 1 + j for j in range(k + 1)])
            pos_ids = torch.tensor([pos_ids_list], device=self.device)
            
            # Build TiDAR Decoding Attention Mask
            dtype = self.model.dtype
            attn_mask = torch.full((B, 1, Q, L + Q), torch.finfo(dtype).min, device=self.device, dtype=dtype)
            
            # Everything sees the prefix (cached)
            attn_mask[:, :, :, :L] = 0.0
            # Last token sees itself
            attn_mask[:, :, 0, L] = 0.0
            # Drafts see Last token + causally themselves
            for i in range(k):
                attn_mask[:, :, 1 + i, L : L + 2 + i] = 0.0
            
            # Mask Blocks
            for i in range(k + 1):
                q_start = 1 + k + i * (k + 1)
                q_end = q_start + (k + 1)
                
                # Sees Last Token
                attn_mask[:, :, q_start:q_end, L] = 0.0
                # Sees accepted drafts up to i
                if i > 0:
                    attn_mask[:, :, q_start:q_end, L + 1 : L + 1 + i] = 0.0
                # Sees itself bidirectionally
                attn_mask[:, :, q_start:q_end, L + q_start : L + q_end] = 0.0

            # Forward Pass (AR + Diffusion all at once)
            outputs = self.model(
                input_ids=model_input,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits # [B, Q, Vocab]

            # 5. Verify Drafts (AR Head)
            ar_logits = logits[:, 0:k+1, :]

            # AR vs Diffusion Blending (Author Variant)
            if trust_ar_ratio is not None and 0.0 <= trust_ar_ratio <= 1.0:
                for i in range(k + 1):
                    # Diffusion prediction corresponding to verification index `i` is at start of block `i`
                    diff_equiv_idx = 1 + k + i * (k + 1)
                    ar_logits[:, i, :] = (
                        trust_ar_ratio * ar_logits[:, i, :] +
                        (1.0 - trust_ar_ratio) * logits[:, diff_equiv_idx, :]
                    )
            
            if logits_processor:
                ar_logits = logits_processor(generated_ids, ar_logits, current_draft)

            # Verification (Greedy vs Probabilistic Rejection Sampling)
            if temperature > 0.0:
                ar_probs = F.softmax(ar_logits / temperature, dim=-1)
                match_count = 0
                next_token = None

                for i in range(k):
                    p = ar_probs[0, i, current_draft[0, i]]
                    q = current_draft_probs[0, i] if current_draft_probs is not None else 1.0

                    if torch.rand(1, device=self.device).item() < (p / q).item():
                        match_count += 1
                    else:
                        # Rejected -> Sample residual to correct
                        if current_draft_probs_full is not None:
                            residual = torch.clamp(ar_probs[0, i] - current_draft_probs_full[0, i], min=0.0)
                        else:
                            residual = ar_probs[0, i]

                        if residual.sum() > 0:
                            residual = residual / residual.sum()
                            next_token = torch.multinomial(residual, 1).unsqueeze(0)
                        else:
                            next_token = torch.multinomial(ar_probs[0, i], 1).unsqueeze(0)
                        break

                if next_token is None:
                    next_token = torch.multinomial(ar_probs[0, k], 1).unsqueeze(0)
            else:
                ar_preds = ar_logits.argmax(dim=-1)
                match_count = 0
                for i in range(k):
                    if ar_preds[0, i] == current_draft[0, i]:
                        match_count += 1
                    else:
                        break
                next_token = ar_preds[:, match_count].unsqueeze(-1)

            total_accepted_drafts += match_count
            new_accepted = current_draft[:, :match_count]
            
            if log_diffusion:
                draft_str = self.tokenizer.decode(current_draft[0].tolist(), skip_special_tokens=True)
                accept_str = self.tokenizer.decode(new_accepted[0].tolist(), skip_special_tokens=True)
                print(f"[TiDAR Step {steps}] Draft: '{draft_str}' | Accepted: '{accept_str}' ({match_count}/{k})")

             # Check stop conditions safely
            added_tokens = torch.cat([new_accepted, next_token], dim=1)[0].tolist()
            stopped = False
            for i, tok in enumerate(added_tokens):
                if tok in stop_tokens:
                    generated_ids = torch.cat([generated_ids, torch.tensor([added_tokens[:i+1]], device=self.device)], dim=1)
                    stopped = True
                    break

            if not stopped:
                generated_ids = torch.cat([generated_ids, new_accepted, next_token], dim=1)

            if stopped or generated_ids.shape[1] >= seq_len + max_new_tokens:
                break

            # Extract the Next Draft (Diffusion Head)
            # We select the Mask Block that corresponds exactly to our match_count
            block_q_start = 1 + k + match_count * (k + 1)
            block_q_end = block_q_start + (k + 1)
            diff_logits = logits[:, block_q_start:block_q_end, :]

            # Slice 1: so the drafts are strictly shifted right and prevent duplication of position L+i+1
            diff_logits_draft = diff_logits[:, 1:, :]
            
            if temperature > 0:
                diff_probs_full = F.softmax(diff_logits_draft / temperature, dim=-1)
                current_draft = torch.multinomial(diff_probs_full.view(-1, diff_probs_full.shape[-1]), 1).view(B, k)
                current_draft_probs = diff_probs_full.gather(dim=-1, index=current_draft.unsqueeze(-1)).squeeze(-1)
                current_draft_probs_full = diff_probs_full
            else:
                current_draft = diff_logits.argmax(dim=-1)

            # KV Cache Eviction
            # Keep: Prefix(L) + Last_Token(1) + Matched_Drafts(match_count)
            keep_len = L + 1 + match_count
            past_key_values.crop(keep_len)
            
            # The corrected token becomes the `Last_Token` for the next loop
            last_token = next_token

        if log_diffusion:
            duration = time.time() - start_time
            gen_tokens = generated_ids.shape[1] - seq_len
            print(f"\n[TiDAR Stats] Generated {gen_tokens} tokens in {duration:.2f}s ({gen_tokens/duration:.2f} t/s).")
            print(f"[TiDAR Stats] Avg Accepted Drafts per step: {total_accepted_drafts/steps:.2f} / {self.draft_len}")

            return generated_ids
