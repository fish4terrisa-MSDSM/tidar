import torch
import math

def build_tidar_hf_mask(B, T, draft_len, device, dtype, base_attention_mask=None):
    """
    Builds a 4D HF-compatible attention mask [B, 1, 2T, 2T].
    0.0 means visible, -inf means masked out.
    Handles base padding mask to prevent attending to <pad> tokens.
    """
    total_len = 2 * T
    mask = torch.full((total_len, total_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    half_T = T

    # 1. Top-Left: Clean -> Clean (Causal)
    causal_mask = torch.tril(torch.ones((half_T, half_T), device=device))
    mask[:half_T, :half_T] = mask[:half_T, :half_T].masked_fill(causal_mask == 1, 0.0)

    # 2. Bottom-Left: Mask -> Clean (Shifted Causal Block)
    num_blocks = math.ceil(half_T / draft_len)
    small_mask_bl = torch.tril(torch.ones((num_blocks, num_blocks), device=device), diagonal=-1)
    expanded_mask_bl = small_mask_bl.repeat_interleave(draft_len, dim=0).repeat_interleave(draft_len, dim=1)
    mask[half_T:, :half_T] = mask[half_T:, :half_T].masked_fill(expanded_mask_bl[:half_T, :half_T] == 1, 0.0)

    # 3. Bottom-Right: Mask -> Mask (Block Diagonal)
    small_mask_br = torch.eye(num_blocks, device=device)
    expanded_mask_br = small_mask_br.repeat_interleave(draft_len, dim=0).repeat_interleave(draft_len, dim=1)
    mask[half_T:, half_T:] = mask[half_T:, half_T:].masked_fill(expanded_mask_br[:half_T, :half_T] == 1, 0.0)

    # Expand to batch dimension
    mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, total_len, total_len).clone()

    # Mask out padding tokens based on input attention mask
    if base_attention_mask is not None:
        # base_attention_mask: [B, T]. Duplicate to [B, 2T] for Clean + Mask segments
        full_base_mask = torch.cat([base_attention_mask, base_attention_mask], dim=1)
        mask = mask.masked_fill(full_base_mask.unsqueeze(1).unsqueeze(2) == 0, torch.finfo(dtype).min)

    # Prevent NaN in Softmax for completely padded query rows
    # by allowing fully masked padding queries to at least attend to themselves
    query_is_pad = (full_base_mask == 0) # [B, 2T]
    diag = torch.eye(total_len, device=device, dtype=torch.bool).unsqueeze(0) # [1, 2T, 2T]
    fix_mask = diag & query_is_pad.unsqueeze(2) # [B, 2T, 2T]

    mask = mask.masked_fill(fix_mask.unsqueeze(1), 0.0)

    return mask
