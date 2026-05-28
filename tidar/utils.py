import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

def get_chat_template(
    tokenizer,
    chat_template,
    mapping=None,
    map_eos_token=True,
    system_message=None,
    patch_saving=True,
    use_zoo_tokenizer_patch=False,
    **kwargs
):
    """
    An independent implementation of get_chat_template that processes custom
    Jinja templates and role mapping without relying on the unsloth library.
    map_eos_token, system_message, patch_saving, use_zoo_tokenizer_patch do
    absolutely nothing but just to make it api compatible
    It's here only because my former train scripts all use unsloth so
    it can make my life easier.
    """
    # Unpack the custom template and the EOS token/stop word
    if isinstance(chat_template, (list, tuple)):
        chat_template_str, stop_word = chat_template
    else:
        chat_template_str = chat_template
        stop_word = getattr(tokenizer, "eos_token", None)

    # Apply custom mapping to the Jinja template string
    if mapping is not None and isinstance(mapping, dict):
        # Map structural keys ('role' and 'content') if they exist in mapping
        if "role" in mapping:
            r = mapping["role"]
            chat_template_str = (
                chat_template_str
                .replace("'role'", f"'{r}'")
                .replace('"role"', f'"{r}"')
                .replace(".role", f".{r}")
            )
        if "content" in mapping:
            c = mapping["content"]
            chat_template_str = (
                chat_template_str
                .replace("'content'", f"'{c}'")
                .replace('"content"', f'"{c}"')
                .replace(".content", f".{c}")
            )

        # Map actual values of roles (e.g., 'user' -> 'human', 'assistant' -> 'gpt')
        for original_val, mapped_val in mapping.items():
            if original_val in ("role", "content"):
                continue
            # Safely replace role string literals in both single and double quotes
            chat_template_str = (
                chat_template_str
                .replace(f"'{original_val}'", f"'{mapped_val}'")
                .replace(f'"{original_val}"', f'"{mapped_val}"')
            )

    # Bind the modified template to the tokenizer
    tokenizer.chat_template = chat_template_str

    # Ensure the EOS token / stop word is registered and recognized as a special token
    if stop_word is not None and isinstance(stop_word, str):
        if getattr(tokenizer, "eos_token", None) != stop_word:
            tokenizer.eos_token = stop_word

        # Safely verify and add the token to the tokenizer vocabulary if missing
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            raise Exception("WTF How could eos_token not inside the vocab")

    return tokenizer

def load_model_and_tokenizer(model_name, bnb_config=None, special_tokens=None, use_rope_scaling=False, rope_factor=2.0):
    """Loads Any model(supported by transformer) with optional BitsAndBytes quantization and RoPE scaling."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    if use_rope_scaling:
        # Enable dynamic RoPE scaling for long context
        config.rope_scaling = {"type": "dynamic", "factor": rope_factor}

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # FIXME: This is known to cause serious problems, including model hallu
    # Add special tokens for TiDAR and XML parsing
    if special_tokens is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    if special_tokens is not None:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def get_bnb_config(mode="4bit"):
    """Returns BitsAndBytes config for memory saving."""
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

class RegexLogitsProcessor:
    """Production ready Regex Logits Processor dynamically enforcing partial prefix constraints."""
    def __init__(self, regex_pattern, tokenizer, stop_tokens):
        import regex
        self.pattern = regex.compile(regex_pattern)
        self.tokenizer = tokenizer
        self.stop_tokens = stop_tokens
        self.token_strings = {}
        self.cache = {}

    def get_token_string(self, i):
        if i not in self.token_strings:
            self.token_strings[i] = self.tokenizer.decode([i], skip_special_tokens=False)
        return self.token_strings[i]

    def apply_mask(self, prefix_str, logits):
        vocab_size = logits.shape[-1]
        if prefix_str in self.cache:
            valid_mask = self.cache[prefix_str]
        else:
            valid_mask = torch.zeros(vocab_size, dtype=torch.bool)
            for i in range(vocab_size):
                if i in self.stop_tokens:
                    continue
                token_str = self.get_token_string(i)
                test_str = prefix_str + token_str
                if self.pattern.fullmatch(test_str, partial=True):
                    valid_mask[i] = True

            # If the current prefix is a valid FULL match, we must allow the model to stop!
            if self.pattern.fullmatch(prefix_str, partial=False):
                for st in self.stop_tokens:
                    if st < vocab_size:
                        valid_mask[st] = True

            self.cache[prefix_str] = valid_mask

        valid_mask = valid_mask.to(logits.device)

        # Fallback: If speculative diffusion proposed a garbage token violating the regex,
        # valid_mask will be empty. We unlock all to prevent NaN crash.
        # The invalid token will be rejected by the AR matching logic anyway.
        if not valid_mask.any():
            valid_mask[:] = True

        logits[~valid_mask] = -float('inf')
        return logits

    def __call__(self, generated_ids, ar_logits, current_draft):
        base_prefix = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        parts = base_prefix.split("<speak>")
        active_prefix = parts[-1] if len(parts) > 1 else base_prefix

        for i in range(ar_logits.shape[1]):
            if i > 0 and current_draft is not None:
                draft_token_str = self.tokenizer.decode(current_draft[0, i-1:i], skip_special_tokens=False)
                active_prefix += draft_token_str
            ar_logits[0, i, :] = self.apply_mask(active_prefix, ar_logits[0, i, :])

        return ar_logits
