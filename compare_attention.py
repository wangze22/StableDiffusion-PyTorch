import torch
from models.multihead_attention import CustomMultiheadAttention


def copy_weights(custom, reference):
    q_w, k_w, v_w = reference.in_proj_weight.chunk(3, dim=0)
    q_b, k_b, v_b = reference.in_proj_bias.chunk(3)

    custom.q_proj.weight.data.copy_(q_w)
    custom.k_proj.weight.data.copy_(k_w)
    custom.v_proj.weight.data.copy_(v_w)
    custom.q_proj.bias.data.copy_(q_b)
    custom.k_proj.bias.data.copy_(k_b)
    custom.v_proj.bias.data.copy_(v_b)
    custom.out_proj.weight.data.copy_(reference.out_proj.weight.data)
    custom.out_proj.bias.data.copy_(reference.out_proj.bias.data)


def run_self_attention_test(embed_dim, num_heads, seq_len, batch_size):
    torch.manual_seed(0)

    reference = torch.nn.MultiheadAttention(
        embed_dim, num_heads, dropout=0.0, batch_first=True
    )
    candidate = CustomMultiheadAttention(
        embed_dim, num_heads, dropout=0.0, batch_first=True
    )
    copy_weights(candidate, reference)

    query = torch.randn(batch_size, seq_len, embed_dim)

    out_ref, attn_ref = reference(query, query, query, need_weights=True)
    out_custom, attn_custom = candidate(query, query, query, need_weights=True)

    print("Self Attention")
    print("  output max diff:", (out_ref - out_custom).abs().max().item())
    print("  attn   max diff:", (attn_ref - attn_custom).abs().max().item())


def run_cross_attention_test(embed_dim, num_heads, query_len, context_len, batch_size):
    torch.manual_seed(0)

    reference = torch.nn.MultiheadAttention(
        embed_dim, num_heads, dropout=0.0, batch_first=True
    )
    candidate = CustomMultiheadAttention(
        embed_dim, num_heads, dropout=0.0, batch_first=True
    )
    copy_weights(candidate, reference)

    query = torch.randn(batch_size, query_len, embed_dim)
    context = torch.randn(batch_size, context_len, embed_dim)

    out_ref, attn_ref = reference(query, context, context, need_weights=True)
    out_custom, attn_custom = candidate(query, context, context, need_weights=True)

    print("Cross Attention")
    print("  output max diff:", (out_ref - out_custom).abs().max().item())
    print("  attn   max diff:", (attn_ref - attn_custom).abs().max().item())


if __name__ == "__main__":
    run_self_attention_test(embed_dim=128, num_heads=8, seq_len=64, batch_size=2)
    run_self_attention_test(embed_dim=64, num_heads=4, seq_len=16, batch_size=4)
    run_cross_attention_test(
        embed_dim=128, num_heads=8, query_len=64, context_len=32, batch_size=3
    )
