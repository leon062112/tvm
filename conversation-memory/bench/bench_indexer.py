"""P0.4 baseline: lightning indexer (fp8 paged MQA logits) via DeepGEMM on H200.

Black-box API-level bench: no kernel internals read (reference-access discipline).
Tensor contract extracted from public API usage in tirx-kernels baseline wrapper:
  q:            fp8_e4m3 [batch, next_n, num_heads, head_dim=128]  -> passed as (q, None)
  fused_kv:     uint8  [num_pages, page_size, 1, head_dim+4]       (fp8 K + per-token fp32 scale)
  weights:      fp32   [batch*next_n, num_heads]
  context_lens: int32  [batch, next_n]
  block_table:  int32  [batch, max_num_pages]
  schedule_meta: deep_gemm.get_paged_mqa_logits_metadata(context_lens, page_size, num_sms, indices)
Output: logits fp32 [batch*next_n, max_num_pages]
"""
import os, sys, json, time
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.cuda.init()
import deep_gemm

PAGE = 64
HEAD_DIM = 128
NUM_SMS = deep_gemm.get_num_sms()

def _make_case(batch, next_n, ctx_pages, num_heads=32, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    max_num_pages = ctx_pages
    num_pages = max(11923, batch * max_num_pages) + 64  # pool larger than any request
    q = torch.randn(batch, next_n, num_heads, HEAD_DIM, device=dev, dtype=torch.bfloat16).clamp_(-2, 2)
    q = q.to(torch.float8_e4m3fn).contiguous()
    kv_bf16 = torch.randn(num_pages, PAGE, HEAD_DIM, device=dev, dtype=torch.bfloat16).clamp_(-2, 2)
    scales = kv_bf16.abs().float().amax(dim=2, keepdim=True).clamp(1e-4) / 448.0
    kv_fp8 = (kv_bf16 / scales).to(torch.float8_e4m3fn)
    fused = torch.empty((num_pages, PAGE, 1, HEAD_DIM + 4), dtype=torch.uint8, device=dev)
    flat = fused.view(num_pages, PAGE * (HEAD_DIM + 4))
    flat[:, : PAGE * HEAD_DIM].copy_(kv_fp8.view(torch.uint8).reshape(num_pages, -1))
    flat[:, PAGE * HEAD_DIM:].copy_(scales.view(torch.uint8).reshape(num_pages, -1))
    fused = fused.contiguous()
    weights = torch.randn(batch * next_n, num_heads, device=dev, dtype=torch.float32).contiguous()
    # context_lens [batch, next_n]: for decode next_n=1 -> fixed ctx; for next_n>1 ragged-ish
    if next_n == 1:
        context_lens = torch.full((batch, 1), ctx_pages * PAGE, dtype=torch.int32, device=dev)
    else:
        base = ctx_pages * PAGE
        context_lens = base - torch.arange(next_n, dtype=torch.int32, device=dev)[None, :]
        context_lens = context_lens.expand(batch, next_n).contiguous()
    # block_table: contiguous identity pages per request
    block_table = torch.arange(batch * max_num_pages, dtype=torch.int32, device=dev)
    block_table = block_table.view(batch, max_num_pages)
    indices = torch.arange(batch * next_n, dtype=torch.int32, device=dev)
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, PAGE, NUM_SMS)
    max_context_len = ctx_pages * PAGE
    return dict(q=q, kv=fused, weights=weights, context_lens=context_lens,
                block_table=block_table, indices=indices, schedule_meta=schedule_meta,
                max_context_len=max_context_len, batch=batch, next_n=next_n,
                ctx=ctx_pages * PAGE, num_heads=num_heads)

def _run(c):
    # Local deep_gemm is an older API: positional args, q as plain Tensor (no tuple),
    # no indices/logits_dtype kwargs.
    return deep_gemm.fp8_paged_mqa_logits(
        c["q"], c["kv"], c["weights"], c["context_lens"], c["block_table"],
        c["schedule_meta"], c["max_context_len"], False)

def bench(c, warmup=5, iters=20):
    for _ in range(warmup):
        _run(c)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); _run(c); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times)//2]  # median ms

if __name__ == "__main__":
    print(f"num_sms={NUM_SMS}")
    results = []
    # decode: next_n=1, B x context
    for batch in [1, 2, 4, 8, 16]:
        for ctx_pages in [64, 160, 512, 1280, 2048]:  # 4K,10K,32K,80K,128K tokens
            try:
                c = _make_case(batch, 1, ctx_pages)
                ms = bench(c)
                res = dict(kind="decode", batch=batch, next_n=1, ctx_tokens=c["ctx"], ms=round(ms, 4))
                results.append(res); print(res, flush=True)
            except Exception as ex:
                print(dict(kind="decode", batch=batch, ctx_pages=ctx_pages, FAIL=repr(ex)[:200]), flush=True)
    # target-verify: next_n=2,4
    for next_n, batch, ctx_pages in [(2, 4, 2048), (4, 8, 512), (2, 1, 512)]:
        try:
            c = _make_case(batch, next_n, ctx_pages)
            ms = bench(c)
            res = dict(kind="tverify", batch=batch, next_n=next_n, ctx_tokens=c["ctx"], ms=round(ms, 4))
            results.append(res); print(res, flush=True)
        except Exception as ex:
            print(dict(kind="tverify", batch=batch, next_n=next_n, FAIL=repr(ex)[:200]), flush=True)
    with open(os.path.join(os.path.dirname(__file__), "indexer_baseline.json"), "w") as f:
        json.dump(results, f, indent=2)
