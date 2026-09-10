# DSA 形状集规格化(锚论文 Table 4 + H200 生产形状)

- **主题**:DSA 复现的目标形状集——组件 A(lightning indexer)/ B(dense MLA decode)/ C(sparse MLA decode)各自的对拍形状与关键超参
- **时间**:2026-09-09(草稿;`[待核实]` 项由 P0.4 对照 DeepSeek 开源 modeling 代码钉死)
- **来源**:论文 Table 4 各行(cake-paper-deep-notes.md)+ 本机 flashinfer 0.4.1 API + DeepSeek V3.2-Exp 公开参数;主计划 [[dsa-reproduction-plan.md]]

## 锚定行(论文原文,数字直接可比)

| 行 | 组件 | 参数 | 论文结果(B200) |
| --- | --- | --- | --- |
| S5 | MQA indexer(=lightning indexer)FP8 | H=32, D=128, Sq=1024, Skv=2048 | 1.270× vs DeepGEMM |
| S6 | Paged MQA indexer FP4 | B=256, H=64, D=128, avgKV=4096, block(page)=64 | 1.0036× |
| S7 | CUTLASS MLA decode(DSv3) | B=128, Skv=4096, page=128 | 1.2174 / 1.1297× |
| S8 | DSv4 sparse MLA decode FP8 | **B=3, ragged query=[3,4,5], H=128, Dqk=Dv=512** | 0.9649×,**论文唯一失败行** |

## 组件 A:lightning indexer(对拍 DeepGEMM)

- 语义:每 query×head group,算 page 级 score(q·k 在 block 上归约)→ top-k page 选择,输出每 query 的 page 索引。
- 关键超参(DeepGEMM lightning indexer 约定):FP8(e4m3)score、page = 64 tokens、stride(索引器 head 对 KV 的抽样密度)。
- 形状族:H=32(论文)/64(生产);Skv 2K–64K tokens;page 64;Sq(新增 query)=1(纯 decode)/32–256。
- `[待核实]`:索引器 head 与 MLA head 的对应方式(head 分组数)、top-k 每 query 块数默认值。

## 组件 B:dense MLA decode(中间门,对拍 flashinfer 0.4.1)

- 语义:page-based KV + FP8(group 反量化)+ qk^T → softmax → pv,decode 增量 token。
- DSv3 MLA 参数:kv_lora_rank=512、qk_nope_head_dim=128、qk_rope_head_dim=64、v_head_dim=128、H(attn heads)∈{128, 64}。
- flashinfer 0.4.1 入口:`flashinfer.decode.BatchDecodeMlaWithPagedKVCacheWrapper`(H200 已装)。
- 形状族:规整 B=1–256、Skv per-request 1K–32K、page 64/128;先规整 B 过门,再 ragged。

## 组件 C:sparse MLA decode(冲刺目标,论文 S8 行)

- 语义:B + indexer 选出的 page 子集;只在这些 page 上做 MLA decode;ragged query(每 request query 数不同)、ragged KV。
- 论文 S8 锚:B=3, query=[3,4,5](即每 request 3/4/5 个 query token), H=128, Dqk=Dv=512, FP8 KV。
- `[待核实]`:page 内 token 数(64/128)、group 反量化 block(通常 128×128 或按 block)、top-k page 数、DSA head 拆分(DSA 用部分 head 做 indexer)。
- 无官方 H200 稀疏基线的话,比较基准 = 组件对拍 + 自建直写 CUDA 臂(P3.3)(见 baselines.md 回退策略)。

## H200 环境常量(写进所有 bench)

- device = sm_90a;BF16 峰值 ~989 TFLOPS?`[待核]`;FP8 ~2× BF16;HBM3e ~4.8 TB/s;132 SM;139.8 GiB。
- L2 5.24e7 B(与 target tag nvidia-h100 一致)。所有微基准报 CUDA 时钟时间,CUPTI GPU span 口径留 P2。

## 用法

- P0.4 用本表形状族跑 baseline microbench → 结果写 baselines.md;
- P1 substrate GEMM 单 shape 去险切片用组件 A/B 的 (M,N,K) 内蕴形状;
- P2 各 kernel 的数值 gate 用本表行;perf gate 对比 baselines.md 数字。
