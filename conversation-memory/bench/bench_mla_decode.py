"""P0.4 baseline: dense MLA decode via FlashMLA (sgl_kernel bundled) on H200.

FlashMLA is the Hopper-native MLA decode kernel (sm_90a). sgl_kernel ships a
compiled copy (`sgl_kernel.flash_mla`), so no source build needed.

Contract (from sglang flashmla_backend.py, public API usage):
  q:            [B, seq, H, kv_lora_rank + qk_rope_head_dim=576]
  k_cache:      [num_pages, PAGE=64, 1, 576]
  block_table:  [B, max_pages] int32
  cache_seqlens:[B] int32
  head_dim_v=512 ; metadata = get_mla_metadata(seqlens, num_q_heads, 1)
  out: [B, seq, H, 512]
"""
import os, json
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.cuda.init()
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

PAGE = 64
CKV, KPE, H, VDIM = 512, 64, 128, 512
D = CKV + KPE  # 576

def _make_case(batch, skv_pages, dtype=torch.bfloat16):
    dev = "cuda"
    npages = batch * skv_pages + 8
    q = torch.randn(batch, 1, H, D, device=dev, dtype=dtype)
    k_cache = torch.randn(npages, PAGE, 1, D, device=dev, dtype=dtype)
    block_table = torch.arange(batch * skv_pages, dtype=torch.int32, device=dev).view(batch, -1)
    cache_seqlens = torch.full((batch,), skv_pages * PAGE, dtype=torch.int32, device=dev)
    meta, num_splits = get_mla_metadata(cache_seqlens, H, 1)
    return dict(q=q, k=k_cache, bt=block_table, sl=cache_seqlens,
                meta=meta, ns=num_splits, batch=batch, skv=skv_pages * PAGE)

def _run(c):
    o, _ = flash_mla_with_kvcache(
        q=c["q"], k_cache=c["k"], block_table=c["bt"], cache_seqlens=c["sl"],
        head_dim_v=VDIM, tile_scheduler_metadata=c["meta"], num_splits=c["ns"],
        softmax_scale=1.0 / (D ** 0.5), causal=True)
    return o

def bench(c, warmup=3, iters=15):
    for _ in range(warmup):
        _run(c)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); _run(c); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts)//2]

if __name__ == "__main__":
    print("FlashMLA dense MLA decode (Hopper/sm90)")
    results = []
    for batch in [1, 2, 4, 16, 64, 128]:
        for skv_pages in [16, 64, 256, 1024]:  # 1K, 4K, 16K, 64K
            try:
                c = _make_case(batch, skv_pages)
                ms = bench(c)
                res = dict(batch=batch, skv_tokens=c["skv"], ms=round(ms, 4))
                results.append(res); print(res, flush=True)
            except Exception as ex:
                print(dict(batch=batch, skv_pages=skv_pages, FAIL=repr(ex)[:200]), flush=True)
    with open(os.path.join(os.path.dirname(__file__), "mla_decode_baseline.json"), "w") as f:
        json.dump(results, f, indent=2)
