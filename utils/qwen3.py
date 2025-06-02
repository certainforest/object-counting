"""
Reversed engineered forward pass for Qwen
- Supports Qwen3
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen3.py
"""
import torch

@torch.no_grad()
def run_qwen3(model, input_ids, attention_mask, return_hidden_states = False):
    """
    Params:
        @model: A model of class `Qwen3ForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        A dictionary with keys:
        - `logits`: The standard B x N x V LM output
        - `attn_out_hidden_states`: If return_hidden_states, a list of length equal to the number of layers, with each element a B x N x D tensor of attention outputs
        - `mlp_out_hidden_states`: If return_hidden_states, a list of length equal to the number of layers, with each element a B x N x D tensor of MLP outputs
        - `layer_out_hidden_states`: If return_hidden_states, a list of length equal to the number of layers, with each element a B x N x D tensor of the final hidden state outputs
    """
    input_embeds = model.model.embed_tokens(input_ids)
    
    cache_position = torch.arange(0, input_embeds.shape[1], device = input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = model.model._update_causal_mask(attention_mask, input_embeds, cache_position, None, None)

    hidden_state = input_embeds
    position_embeddings = model.model.rotary_emb(hidden_state, position_ids)

    attn_out_list = []
    mlp_out_list = []
    layer_out_list = []
        
    for layer_ix, layer in enumerate(model.model.layers):
        # SA
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        attn_out, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask, position_ids = position_ids, position_embeddings = position_embeddings)
        hidden_state = residual + attn_out
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)
        
        mlp_out = layer.mlp(hidden_state)
        hidden_state = residual + mlp_out

        if return_hidden_states:
            attn_out_list.append(attn_out.detach())
            mlp_out_list.append(mlp_out.detach())
            layer_out_list.append(hidden_state.detach())

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)
    return {'logits': logits, 'attn_out_hidden_states': attn_out_list, 'mlp_out_hidden_states': mlp_out_list, 'layer_out_hidden_states': layer_out_list}