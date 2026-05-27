"""Phase 4 integration: plug the kernels built in Phase 1-3 into HF Llama.

Each submodule is one kernel's monkeypatch surface:

  attention_patch  : Phase 1 v3 decode_attention into LlamaSdpaAttention's
                     F.scaled_dot_product_attention boundary.
  kv_int4_patch    : Phase 2 INT4 KIVI KV cache (DynamicCache replacement). (4b)
  w4a16_patch      : Phase 3 W4A16 GEMM into nn.Linear forward. (4c)

Each patch is implemented as a context manager (`with patched_<thing>():`) so
the eval scripts can A/B vanilla vs patched cleanly in the same process.
"""
