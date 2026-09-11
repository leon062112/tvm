# P1 进展:WGMMA substrate(进行中,卡在 tensormap)

- **主题**:P1 首次动手——recon 完成、CUDA 构建修复、一个真实代码 bug 修复、去险脚本就位;当前阻塞在编译期 `Unknown stack alloca type tensormap`
- **时间**:2026-09-11
- **来源**:本轮开发;上游 [[dsa-reproduction-plan.md]](P1 定义)、[[tirx-wgmma-gap.md]](缺口分析)、[[h200-env-notes.md]](环境)

## 状态一句话

P1 **进行中**:wgmma 脚本层表面其实已存在(推翻了原计划里"要新写 intrinsic"的假设),走通了编译前几步并修掉一个预埋 bug;离"第一个 bit-accurate BF16 WGMMA"还差解决一个编译期 tensormap 分配问题。

## 关键结论(修正原计划)

1. **P1.2 新 intrinsic 不需要**:wgmma 表面已在 script 层齐备——
   - `T.ptx["wgmma.mma_async.sync.aligned.m{M}n{N}k{K}.f32.bf16.bf16"](d_regs..., descA, descB, scale_d, 1, 1, transA, transB)`(路由 ptx 表 `table.py:7166+` 的 `wgmma_bf16_ss`);
   - `T.cuda.wgmma.encode_matrix_descriptor(addr, addr, ldo, sdo, swizzle)`(`op.py:705`);
   - `T.ptx.wgmma.fence/commit_group/wait_group`(`table.py:7420+`);
   - `T.cuda.wgmma.noop_barrier(reg)`(寄存器 pin)。
   - 现成可运行样例:`tests/python/tirx/codegen/test_codegen_hopper.py::test_wgmma_ss_nt`(M=64,N=64,K=16,fp16→f32,swizzle=3,128 线程 warpgroup;encode args `[ldo=1,sdo=64,swizzle=3]`)。
2. 所以 P1 的真正产出收敛为 **P1.3 dispatch 变体 `gemm_async → wgmma`**(对标 `gemm_async/tcgen05.py`,但累加器在寄存器、无 TMEM/FP8/SF),不再需要新 intrinsic。

## 本次完成

- **recon**:`GemmAsync` op 契约(args: C,A,B,transA,transB,accum / 或 +SFA,SFB);tcgen05 变体的入口/scope 检查/描述子编码范式;ptx 表 wgmma 条目(ss/rs 两种 form,操作数顺序 d, a_desc, b_desc, scale_d, scale_a, scale_b, trans_a, trans_b)。
- **CUDA 构建修复**:见 [[h200-env-notes.md]]——`USE_CUDA` 必须写进 `build/config.cmake`,否则 `-DUSE_CUDA=ON` 被 config.cmake 的 `set(USE_CUDA OFF)` 覆盖。修复后 `has_cuda()=True`、`cc=(9,0)`。
- **修掉一个预埋 bug**(本仓库首个 P1 源码改动):
  - `python/tvm/backend/cuda/cpp/asm.py:184` —— `codegen_cuda_wgmma_noop_barrier` 用 `reg.dtype`,但 tirx `BufferLoad` 没有 `.dtype`(规范访问器是 `.ty`,见 `op.py:135` 等)。
  - 改为 `dtype = str(reg.ty)`。**尚未端到端验证**(被下游阻塞,见下)。
- **去险脚本**:`/root/code/dsa-bench/wgmma_gemm_bf16.py`(已存档 [[bench/wgmma_gemm_bf16.py]]),单 tile BF16 M=64,N=64,K=16,对拍 torch。

## 当前阻塞

编译期报错:
```
tvm.error.InternalError: Unknown stack alloca type tensormap
  src/target/source/codegen_c_host.cc:304  CodeGenCHost::VisitExpr_(CallNode)
```
来源:测试/脚本里的 `T.tvm_stack_alloca("tensormap", 1)` + `T.call_packed("runtime.cuTensorMapEncodeTiled", ...)`(host 侧建 TMA tensormap)。

- `tensormap` 在源码中仅出现于:`src/target/source/codegen_c.cc`、`src/target/build_common.h`、`src/target/llvm/codegen_{llvm,cpu}.cc`、`src/tirx/transform/lower_tirx_dedup_tensormap.cc`——**`codegen_c_host.cc` 没有 `tensormap` 类型的分支**。
- `[待查]`:是构建/配置缺项(某 USE_* 选项),还是该 host codegen 路径确实未支持 tensormap 栈分配。这是下一步的第一件事。

## 测试基建坑(已确认)

`tests/python/tirx/codegen/conftest.py::pytest_collection_modifyitems` 在 `not has_cuda_compute(10)` 时把该目录**所有 `gpu` 标记测试强制 skip** → H200(cc9.0)上即便 `test_wgmma_ss_nt` 自身门控是 cc==9.0 也跑不了。**绕法:独立脚本**(如 `bench/wgmma_gemm_bf16.py`),不经该 conftest。

## 下一步(P1 剩余)

1. 解决 `tensormap` 栈分配(或改走不用 tensormap 的 A/B 加载路径,先拿数值正确性)。
2. 跑通 BF16 单 tile → 与 torch 对拍,拿**第一个 bit-accurate WGMMA**(T1 核心)。
3. 扩到 K 循环(多 K-tile,描述子随 K 偏移)+ 多 shape → 对拍 cuBLAS(810 TFLOPS 基线)。
4. 实现 **P1.3 dispatch 变体** `gemm_async → wgmma`(寄存器累加器,bf16 起步)。
5. 首个通过 kernel 入语料台账(第 1 条)。

[[dsa-reproduction-plan.md]] [[baselines.md]] [[h200-env-notes.md]] [[tirx-wgmma-gap.md]]
