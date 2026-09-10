# DSA 基线 survey + microbench(H200)

- **主题**:DSA 各组件在 H200 上的可用基线(black-box)与已测性能数字;不可用的记录阻塞原因与回退
- **时间**:2026-09-09(P0.4);bench 脚本在 `/root/code/dsa-bench/`,原始数字 `indexer_baseline.json`
- **来源**:本机实测;形状定义见 [[shapes.md]];主计划 [[dsa-reproduction-plan.md]]

## 环境前提

- 机器:sm_90a H200 级,132 SM,139.8 GiB,PCIe,CUDA 12.8,driver 570.148.08。**以 torch/pynvml 判 cc(9.0),勿信 nvidia-smi 的 compute_cap 列(掩码 8.9)**。
- bench 口径:CUDA Event 计时,warmup 5 / iter 20,取中位数,ms。CAKE 级 CUPTI GPU-span + L2 flush 留 P2。

## 组件 A:lightning indexer(fp8 paged MQA logits)

- **基线 = deep_gemm**(预装,sm90 可用;`fp8_paged_mqa_logits`,旧 API 位置参数、无 `indices`/`logits_dtype` kwargs;`get_paged_mqa_logits_metadata(context_lens, page_size, num_sms)`)。
- 张量契约(P2 复现锚点,从 deep_gemm 测试文件提取):q fp8_e4m3 `[B, next_n, H, D]`;fused_kv_cache uint8 `[num_pages, page, 1, D+4]`(fp8 K + 末尾 4B/token scale);weights fp32 `[B*next_n, H]`;context_lens int32 `[B]`;block_table int32 `[B, max_pages]`;输出 logits fp32 `[B*next_n, max_context_len]`,页外为 -inf。
- **限制**:本地版本 `next_n` 只支持 1 或 2(next_n=4 断言失败)——tverify 覆盖受限,记 `[待新版]`。

### 已测数字(H200,page=64,D=128,H=32,logits fp32)

| batch | ctx=4K | 10K | 32K | 80K | 128K |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0234 | 0.0240 | 0.0237 | 0.0233 | 0.0239 |
| 2 | 0.0237 | 0.0236 | 0.0237 | 0.0236 | 0.0242 |
| 4 | 0.0235 | 0.0237 | 0.0236 | 0.0249 | 0.0366 |
| 8 | 0.0237 | 0.0239 | 0.0238 | 0.0466 | 0.0591 |
| 16 | 0.0331 | 0.0332 | 0.0421 | 0.0616 | 0.0859 |

(单位 ms;小 batch 被 launch 常数主导 ≈0.023ms。target-verify: next_n=2 B=4 ctx=131K → 0.0431ms。)

## 组件 B:dense MLA decode(中间门)

**当前无可用 H200 基线**,三条路全断,逐一记录:

| 候选 | 结果 | 阻塞 |
| --- | --- | --- |
| flashinfer 0.4.1 `BatchDecodeMlaWithPagedKVCacheWrapper` | ❌ | `MLA decode kernel is not supported on this GPU (SM90). Supported architecture: SM80.`(该构建只有 SM80 fallback) |
| flashinfer `trtllm_batch_decode_with_kv_cache_mla` | ❌ | 需从 NVIDIA artifactory 下载 cubin(`edge.urm.nvidia.com` 超时失败)——网络环境限制 |
| sgl_kernel `cutlass_mla_decode`(sglang 自带) | ❌ | **`cutlass_mla_decode is only supported on compute capability 10.0, but found sm version 90`** —— CUTLASS MLA decode 是 SM100-only,印证它用 tcgen05 |
| DeepSeek FlashMLA(github 源码) | ❌ | sm100 需 NVCC 12.9(本机 12.8)→ `FLASH_MLA_DISABLE_SM100=1` 绕过;但 **gcc 10.2.1 编 CUTLASS 头 segfault(cc1plus)**;conda 装 gcc-13 因 channel URL 被代理剥冒号(`https//`)失败 |

**结论:H200 上 dense MLA decode 无现成可用基线**。这正是 CAKE Table 4 S7(CUTLASS MLA decode)在论文里是 SM100-only 的侧写——CUTLASS MLA decode 上 Blackwell 才有,SM90 的 dense MLA 好基线实际是 FlashMLA,而它被 gcc 卡死。

- **回退策略(P0.4 决策)**:dense decode 的**数值 oracle** 用 PyTorch 参考(DeepSeek modeling 级,允许);**perf 基线**优先 FlashMLA(gcc13 成功即补数字);若 gcc13 仍不行,用 flashinfer GQA 通用 decode(`BatchDecodeWithPagedKVCacheWrapper`)在相同 (B,SKV,heads) 上做量级参照(非 MLA 语义,仅校准 perf 数量级),并在 P2 以 tirx 自建 kernel 互相校验。
- DSv3 MLA 参数(形状集锚):ckv=512、kpe=64、heads=128、page=64。

## 组件 C:sparse MLA decode(冲刺目标)

- 官方/社区在 H200 上**无现成基线**(CAKE S8 论文唯一失败行,参考实现 8942+9125 LOC 不开源)。**默认走回退路径**:T2 gate 以"数值正确 vs PyTorch oracle"为硬门槛,perf 以"tirx 自建 vs P3.3 直写 CUDA 臂"及 vs dense 上界为参照。

## P1 前置:GEMM 基线(substrate 对拍用)

- **cuBLAS BF16 GEMM**:4096×4096×8192 → 0.339 ms ≈ **810 TFLOPS**(H200 BF16 理论峰值 ~989 TFLOPS,已用 ~82%)。P1 gate 的 WGMMA 正确性/性能对拍以此 + torch 为参照。
- nvidia-cutlass-dsl 4.2.1 import 可用,可作 CUTLASS 级参照。

## P0 gate 判断

- ✅ 构建 + 冒烟过;✅ indexer 基线数字落地;✅ 测试面摸底(下)。
- ⚠️ dense decode 基线工具链阻塞,走回退,不阻塞进 P1(P1 是 WGMMA GEMM substrate,与 decode 基线无关)。

## H200 tirx 测试面(P0.2 摸底)

- 全树 `pytest tests/python/tirx --collect-only`:2709 项,1 个收集错(registry 需 Blackwell,预期)。
- **门控结构**:`tests/python/tirx/operator/tile_primitive/cuda/conftest.py` 用 `env.has_cuda_compute(10)` 把整个 CUDA tile-primitive 套件 skip 到 cc≥10;codegen 里也有 cc 条件 → **H200(cc9.0)上大量 GPU 测试结构性 skip,这是记忆"语料基质缺失"的直接实证**。
- H200 实测可跑且过:编译器侧 `test_layout/test_exec_scope/test_verifier/test_exec_context` 96 全过;`operator/` 143 passed / 1564 skipped;`codegen/` 154 passed / 320 skipped / **9 failed**(见下)。
- 9 failed 全部 PTX-dialect/tensor-map 生成侧(test_codegen_cuda 2 项 + test_ptx_dialect 7 项),**我们未改任何源码,属预存在**;疑似与 cc9.0 路径或预存在 bug 相关,**记 [待查],不进 P1 关键路径**。分类(按 SKILL.md):B 类候选,但均不触及我们将写的 wgmma/dispatch 代码,先记录。
- 结论:**H200 上 tirx 自身回归只能覆盖编译器侧 + 部分 codegen;GPU kernel 正确性语料要靠自己从第 1 个 WGMMA 开始积累**(与 [[tirx-harness-gap.md]] 卡点②一致)。

[[dsa-reproduction-plan.md]] [[shapes.md]] [[tirx-harness-gap.md]] [[tirx-cake-reproduction.md]]
