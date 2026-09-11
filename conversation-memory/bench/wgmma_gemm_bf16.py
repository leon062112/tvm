"""P1 de-risk: single-tile BF16 WGMMA GEMM on H200, validated vs torch.

Modeled on tests/python/tirx/codegen/test_codegen_hopper.py::test_wgmma_ss_nt
(script-level wgmma surface: T.ptx[chain] + T.cuda.wgmma.encode_matrix_descriptor
+ fence/commit/wait_group). bf16/f16 share the same f32-accumulator fragment
layout, so only dtype changes. TMA + 128B swizzle (swizzle=3) path included.

Semantics: C = A @ B, A [M,K], B [K,N], C [M,N] (all matched-buffer order);
transA=transB=True in the test means A stored [K,M], B stored [K,N].
"""
import numpy as np
import torch
import tvm
from tvm.script import tirx as T
import math

def make_wgmma(M=64, N=64, K=16, in_dtype="bfloat16", out_dtype="float32",
               transA=True, transB=True, swizzle=3):
    shapeA = (M, K) if not transA else (K, M)
    shapeB = (N, K) if not transB else (K, N)
    shapeC = (M, N)
    elem_bytes = tvm.DataType(in_dtype).bits // 8
    coordA = [0, 0]; coordB = [0, 0]
    A_bytes = elem_bytes * math.prod(shapeA)
    B_bytes = elem_bytes * math.prod(shapeB)
    C_elems = math.prod(shapeC) // 128
    ptx_ab = {"float16": "f16", "bfloat16": "bf16"}[in_dtype]
    ptx_d = {"float32": "f32", "float16": "f16"}[out_dtype]
    chain = f"wgmma.mma_async.sync.aligned.m{M}n{N}k{K}.{ptx_d}.{ptx_ab}.{ptx_ab}"

    A_outer, A_inner = shapeA
    A_tma_args = [A_inner, A_outer, A_inner * elem_bytes, A_inner, A_outer, 1, 1, 0, swizzle, 0, 0]
    B_outer, B_inner = shapeB
    B_tma_args = [B_inner, B_outer, B_inner * elem_bytes, B_inner, B_outer, 1, 1, 0, swizzle, 0, 0]
    enc = [1, 64, swizzle]

    @T.prim_func
    def main(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle):
        A = T.match_buffer(A_ptr, shapeA, dtype=in_dtype, align=16)
        B = T.match_buffer(B_ptr, shapeB, dtype=in_dtype, align=16)
        C = T.match_buffer(C_ptr, shapeC, dtype=out_dtype, align=16)
        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, in_dtype, len(shapeA), A.data, *A_tma_args)
        B_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, in_dtype, len(shapeB), B.data, *B_tma_args)
        T.device_entry()
        tx = T.thread_id([128])  # warpgroup = 128 threads
        A_smem = T.alloc_buffer(shapeA, in_dtype, scope="shared", align=1024)
        B_smem = T.alloc_buffer(shapeB, in_dtype, scope="shared", align=1024)
        bar = T.shared_scalar("uint64")
        phase: T.int32
        descA: T.uint64
        descB: T.uint64
        C_local = T.alloc_buffer((C_elems,), out_dtype, scope="local")
        phase = 0
        if tx == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(bar), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if tx == 0:
            T.ptx[f"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                A_smem.data, T.address_of(A_map), *coordA, T.address_of(bar))
            T.ptx[f"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                B_smem.data, T.address_of(B_map), *coordB, T.address_of(bar))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(bar), T.uint32(A_bytes + B_bytes))
        T.cuda.mbarrier_wait(T.address_of(bar), phase)
        T.cuda.cta_sync()
        for i in T.serial(0, C_elems):
            C_local[i] = T.Cast(out_dtype, 0.0)
            T.cuda.wgmma.noop_barrier(C_local[i])
        T.cuda.wgmma.encode_matrix_descriptor(T.address_of(descA), A_smem.data, *enc)
        T.cuda.wgmma.encode_matrix_descriptor(T.address_of(descB), B_smem.data, *enc)
        T.ptx.wgmma.fence.sync.aligned()
        T.ptx[chain](*[C_local[i] for i in range(C_elems)], descA, descB,
                     0, 1, 1, int(transA), int(transB))
        T.ptx.wgmma.commit_group.sync.aligned()
        T.ptx.wgmma.wait_group.sync.aligned(0)
        for i in T.serial(0, C_elems):
            T.cuda.wgmma.noop_barrier(C_local[i])
        for i in T.serial(0, C_elems // 4):
            row = T.meta_var((tx % 32) // 4 + (tx // 32) * 16)
            col = T.meta_var(i * 8 + tx % 4 * 2)
            C[row, col] = C_local[i * 4]
            C[row, col + 1] = C_local[i * 4 + 1]
            C[row + 8, col] = C_local[i * 4 + 2]
            C[row + 8, col + 1] = C_local[i * 4 + 3]
    return main, shapeA, shapeB, shapeC, in_dtype, out_dtype

if __name__ == "__main__":
    M, N, K = 64, 64, 16
    func, shapeA, shapeB, shapeC, in_dtype, out_dtype = make_wgmma(M, N, K)
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=tvm.target.Target("cuda"), tir_pipeline="tirx")

    torch.manual_seed(0)
    A_t = torch.randn(shapeA, device="cuda").to(torch.bfloat16)
    B_t = torch.randn(shapeB, device="cuda").to(torch.bfloat16)
    C_ref = (A_t.t().float() @ B_t.float()).reshape(shapeC)  # A stored [K,M], B [K,N]

    dev = tvm.cuda(0)
    A_tvm = tvm.runtime.tensor(A_t.cpu().numpy().astype(in_dtype), device=dev)
    B_tvm = tvm.runtime.tensor(B_t.cpu().numpy().astype(in_dtype), device=dev)
    C_tvm = tvm.runtime.tensor(np.zeros(shapeC).astype(out_dtype), device=dev)
    mod(A_tvm, B_tvm, C_tvm)
    got = torch.from_numpy(C_tvm.numpy())
    diff = (got - C_ref.cpu()).abs()
    print(f"BF16 WGMMA {M}x{N}x{K}: max abs diff = {diff.max().item():.6g}")
    print("PASS" if torch.allclose(got, C_ref.cpu(), rtol=1e-2, atol=1e-2) else "FAIL")
