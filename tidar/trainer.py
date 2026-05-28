# Hook isinstance to trick accelerate and trl to treat our TiDARModel as
# a PeftModel
# ~~ Trick or Treat :D ~~
import builtins

# Save the original isinstance function
_orig_isinstance = builtins.isinstance

def _tidar_isinstance(obj, class_or_tuple):
    try:
        # Check if the object is an instance of TiDARModel (using string check to avoid circular imports)
        if type(obj).__name__ == "TiDARModel":
            # Only trigger if the wrapped base model is configured with PEFT
            if getattr(obj, "_hf_peft_config_loaded", False):
                from peft import PeftModel
                if class_or_tuple is PeftModel:
                    return True
                if _orig_isinstance(class_or_tuple, tuple) and PeftModel in class_or_tuple:
                    return True
    except Exception:
        pass
    return _orig_isinstance(obj, class_or_tuple)

# Override the built-in isinstance globally
builtins.isinstance = _tidar_isinstance

from trl import SFTTrainer, DPOTrainer
import torch
import torch.nn.functional as F
import inspect

class TiDARSFTTrainer(SFTTrainer):
    # Standard SFTTrainer works out of the box because our TiDARModel returns (loss, logits)
    pass

class TiDARDPOTrainer(DPOTrainer):
    def _get_tidar_batch_logps(self, logits, labels, is_shifted=False):
        """
        Computes the log probabilities of the given labels under the given logits.
        If is_shifted=False, we manually perform causal shifting (AR).
        If is_shifted=True, we keep them unshifted (Diffusion).
        """
        if not is_shifted:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]

        loss_mask = labels != -100

        # Clamp labels temporarily to prevent indexing out of bounds on ignored pad tokens
        clamped_labels = labels.clone()
        clamped_labels[clamped_labels == -100] = 0

        #"""
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=clamped_labels.unsqueeze(2)
        ).squeeze(2)

        # Zero out the logps on masked elements before summing
        return (per_token_logps * loss_mask).sum(-1)

    def _manual_dpo_concat(self, batch):
        chosen_ids = batch["chosen_input_ids"]
        rejected_ids = batch["rejected_input_ids"]
        chosen_labels = batch.get("chosen_labels", chosen_ids.clone())
        rejected_labels = batch.get("rejected_labels", rejected_ids.clone())

        # Pull or default mask
        chosen_mask = batch.get("chosen_attention_mask", torch.ones_like(chosen_ids))
        rejected_mask = batch.get("rejected_attention_mask", torch.ones_like(rejected_ids))

        max_length = max(chosen_ids.shape[1], rejected_ids.shape[1])
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) if getattr(self, "tokenizer", None) is not None else 0
        if pad_token_id is None: pad_token_id = 0
        label_pad_id = getattr(self, "label_pad_token_id", -100)

        def pad_tensor(tensor, max_len, pad_value):
            if tensor.shape[1] < max_len:
                pad = torch.full((tensor.shape[0], max_len - tensor.shape[1]), pad_value, dtype=tensor.dtype, device=tensor.device)
                return torch.cat([tensor, pad], dim=1)
            return tensor

        input_ids = torch.cat([pad_tensor(chosen_ids, max_length, pad_token_id), pad_tensor(rejected_ids, max_length, pad_token_id)], dim=0)
        labels = torch.cat([pad_tensor(chosen_labels, max_length, label_pad_id), pad_tensor(rejected_labels, max_length, label_pad_id)], dim=0)
        attention_mask = torch.cat([pad_tensor(chosen_mask, max_length, 0), pad_tensor(rejected_mask, max_length, 0)], dim=0)

        return input_ids, labels, attention_mask

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        """Overrides DPO loss calculation to include TiDAR logic."""
        # Unpack standard DPO batch
        policy_chosen_logits, policy_rejected_logits, policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(model, batch)
        
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    _, _, ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(model, batch)
            else:
                _, _, ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(self.ref_model, batch)

        # Standard DPO Loss calculation
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        
        loss = -F.logsigmoid(self.beta * logits).mean()
        
        reward_accuracies = (logits > 0).float()

        return loss, policy_chosen_logps.mean().detach(), policy_rejected_logps.mean().detach(), reward_accuracies.mean().detach()

    def concatenated_forward(self, model, batch):
        """Custom forward to calculate logprobs for both AR and Diffusion parts."""
        # Robustly extract or build concatenated inputs regardless of TRL version
        if "chosen_input_ids" in batch and "rejected_input_ids" in batch:
            input_ids, labels, attention_mask = self._manual_dpo_concat(batch)
        elif "concatenated_input_ids" in batch:
            input_ids = batch["concatenated_input_ids"]
            labels = batch.get("concatenated_labels", batch.get("labels", input_ids.clone()))
            attention_mask = batch.get("concatenated_attention_mask", None)
        elif "input_ids" in batch:
            # Batch is already concatenated by a newer data collator
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids.clone())
            attention_mask = batch.get("attention_mask", None)
        elif hasattr(self, "concatenated_inputs"):
            try:
                sig = inspect.signature(self.concatenated_inputs)
                kwargs = {}
                if "is_encoder_decoder" in sig.parameters:
                    kwargs["is_encoder_decoder"] = getattr(self, "is_encoder_decoder", False)
                if "label_pad_token_id" in sig.parameters:
                    kwargs["label_pad_token_id"] = getattr(self, "label_pad_token_id", -100)
                if "padding_value" in sig.parameters:
                    kwargs["padding_value"] = getattr(self, "padding_value", 0)
                if "device" in sig.parameters:
                    kwargs["device"] = getattr(getattr(self, "accelerator", None), "device", None)

                concatenated_batch = self.concatenated_inputs(batch, **kwargs)
                input_ids = concatenated_batch.get("concatenated_input_ids", concatenated_batch.get("input_ids"))
                labels = concatenated_batch.get("concatenated_labels", concatenated_batch.get("labels"))
                attention_mask = concatenated_batch.get("concatenated_attention_mask", None)
            except Exception:
                input_ids, labels, attention_mask = self._manual_dpo_concat(batch)
        else:
            input_ids, labels, attention_mask = self._manual_dpo_concat(batch)
        
        # Do NOT pass labels=labels here. DPO does not use standard Cross-Entropy.
        # Passing labels triggers TiDARModel's standard cross-entropy calculation which massively blows up VRAM.
        # We only need the raw logits to calculate our DPO logps below!
        outputs = model(input_ids, attention_mask=attention_mask, output_full_logits=True) # TiDARModel handles the internal concatenation
        logits = outputs.logits
        B, S = input_ids.shape

        # AR Causal Part: Pass unshifted inputs, helper handles shifting
        ar_logits = logits[:, :S, :]
        ar_labels = labels.clone()
        ar_logps = self._get_tidar_batch_logps(ar_logits, ar_labels, is_shifted=False)

        # Diffusion Part: Pass directly as unshifted
        diff_logits = logits[:, S:, :]
        diff_labels = labels.clone()
        diff_logps = self._get_tidar_batch_logps(diff_logits, diff_labels, is_shifted=True)

        total_logps = ar_logps + diff_logps # Combine signals

        # Split back to chosen / rejected
        if "chosen_labels" in batch:
            len_chosen = batch["chosen_labels"].shape[0]
        elif "chosen_input_ids" in batch:
            len_chosen = batch["chosen_input_ids"].shape[0]
        else:
            len_chosen = B // 2
        chosen_logps = total_logps[:len_chosen]
        rejected_logps = total_logps[len_chosen:]
        
        return logits[:len_chosen], logits[len_chosen:], chosen_logps, rejected_logps
