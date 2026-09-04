# CAKE 论文深读笔记(cake.pdf 全文)

- **主题**:cake.pdf 21 页全文深读,提取正文/附录中此前未记录的细节(设计原则、架构矩阵、已知 kernel 对照表、泛化阶段、语料流程、预算量级)
- **时间**:2026-09-04
- **来源**:`cake.pdf`(arXiv 2608.12629)pymupdf 全文提取;对照 [[cake-paper.md]](概要),本文件是细节补充

## 设计原则 P1–P8(附录 B.1,复现时的验收标准)

P1 Ergonomic(NumPy/PyTorch 用户熟悉的编辑模型)/ P2 Performance-transparent(性能相关硬件决策可见)/ P3 Canonical(每操作一个规范形式)/ P4 Statically type-checked(构造期拒绝病态程序)/ P5 Analysis-friendly(IR 暴露静态分析所需信息)/ P6 Test-gated(IR 改动过语料回归)/ P7 Analysis-consistent(数据模型改动必伴随分析更新)/ P8 Hardware-grounded(每个操作文档化硬件行为)。P5+P7 保证新原语可分析,P6 防回归。引用 Tao「proof abundance」:瓶颈从生成转向验证与理解。

## 能力四类 + 资源五类(B.2/B.3)

- 四类操作词汇:Compute / Memory movement(含异步与集合传输)/ Synchronization / Control and scheduling(warp 角色、pipeline 管理、persistent、多块协调)。
- 五类声明式资源:smem regions、tmem regions、sync objects、warp roles、pipelines——声明记录 shape/ownership/lifetime;barrier 地址、phase bit、TMEM offset、描述符编码、warp identity 等机械元数据由 lowering 推导,不由 agent 手写。

## 架构矩阵 Table 3(B.5)——复现的硬件边界

| target | tensor-core 路径与特性 |
| --- | --- |
| sm_80 A100 | mma.sync, ldmatrix, cp.async;无 TMA/cluster/TMEM |
| sm_89 | 同上 + FP8 张量核 |
| **sm_90a H100/H200** | **WGMMA + mma.sync;TMA、thread-block clusters、async barriers、DSM** |
| sm_100a B200 | tcgen05.mma + TMEM;cta_group::2 |
| sm_103a B300 | + tcgen05.ld.red、K=96 block-scaled MMA |
| sm_120a/121a | mma.sync 系 + TMA/cluster/DSM;无 tcgen05/TMEM |

关键:CAKE 官方 sm_90a 路径就是 WGMMA,直接印证 [[tirx-wgmma-gap.md]] 的缺口定性。编译器要求精确 target 匹配、拒绝静默降级。**Timing model 只在 B200(measured)与 H100(独立校准)可用,其他 target 明确报 coverage limitation 而非继承估计**——H200 复现必须自校准。

## cost model 的四句散点(论文对内部的全部透露)

1. §3.1:calibrated cost model 返回 bottleneck attribution + optimization guidance(结构化报告,非单标量);
2. §2.3:只在有 target 特定校准处给性能估计;
3. B.5:timing-model 覆盖 evidence-gated、逐 target 独立;
4. §3.2:系统性误预测 → 变成 calibration target(经验校准、随 harness 进化修正)。
App.C:静态分析与性能模型**故意不完备**,只 rank/filter,GPU 实测为 ground truth,允许误报漏报。设计含义见 [[tirx-harness-gap.md]]。

## Fig.3 Cake IR 语法要点(FMHA 片段,复现表面层对照)

`@cake.schedule()`;`lm.smem(98304)`;`pool.view(offset, shape, dtype, stage=3)`;`lm.tmem(cols, width, shape, dtype)`;`lm.role(warps=[0])`;`lm.pipeline(stages=3)`;`lm.barrier(count, prod=[load], cons=[mma], init_count, pipeline)`;`with load:` / `with mma:`;`smem_q.tma_load(coords, stage, barrier)`;`lm.wait/fence_proxy/mma(init=...)`。逐构造映射 tirx 见 [[tirx-cake-ir-feasibility.md]]。

## Table 4 已知 kernel 对照(附录 E,11 项 10 达标)

- 最强:**MQA indexer FP8 1.270× / FP4 1.273×**(S5: H=32, D=128, Sq=1024, Skv=2048;参考 DeepGEMM,dev 704 LOC);Paged MQA indexer FP4 1.0036(S6: B=256, H=64, D=128, avgKV 4096, block 64)——MQA indexer 即 DeepSeek DSA 的 lightning indexer。
- **唯一未达标:DSv4 sparse MLA decode FP8 0.9649**(S8: DeepSeek-V4 sparse MLA,B=3,ragged query [3,4,5],H=128,Dqk=Dv=512;参考 8942 dev + 9125 含支撑 LOC vs Cake 1393)——即 DSA decode 路径,说明 sparse MLA decode 是最难啃的一行。
- 其余:FA4 fwd 1.0045 / bwd 1.0470(S1);TRTLLM GQA decode 1.043(S2);1D1D FP8 GEMM 1.037(S3);masked grouped GEMM 1.0174(S4);CUTLASS MLA decode 1.2174 / 1.1297(S7: DSv3, B=128, Skv=4096, page 128)。
- 论文自评:低于参考是编译器集成成熟度问题(用最近可用策略顶替),非算法差距;indexer 超额来自 porting 中搜到参考实现没有的优化。Cake IR LOC 全面短于参考。

## Section 6 泛化阶段(独立阶段,不是"多跑几个 shape")

单 shape 内环 vs dispatcher-inclusive 外环:不同目标、不同排序信号、不同失败模式;错误或过慢的 seed 打回内环。防泄漏:先声明有效 shape 域,dispatcher 谓词不得引入新评估行,held-out shards 验证。portfolio 实测(GB200 vs FlashLib 0.2.0):KNN build 8 族/80 路由/112 shape Gspan 1.418;KNN search 198 shape 2.116;KMeans 12 路由/124 shape 1.803。一路由 = 一独立 Cake IR 程序,可独立分析/benchmark;策略是尽量复用同一物理 schedule,仅当域要求实质改动才加路由。

## App A 语料流程 + 预算量级

- IR 出生方式五步:语料收集(生产 kernel,他 DSL 先由 agent 翻成 CUDA+PTX)→ 抽象提取(barrier choreography / pipeline staging / warp-role / TMA descriptor / TMEM 生命周期)→ 硬件导向设计(Blackwell 模型偏置)→ 原则验证(P1–P8)→ port 驱动扩张。**语料是产物不是前提**——bootstrap 我们 harness 语料的直接依据。
- 预算量级:Flash-KMeans clean-start 80M token/轮 × 3 轮/臂;KDA 两阶段累计约 1200M token(Fig.6);TinyGEMM 四 shape ~100M+;Alpha-MoE ~65M。模型固定 GPT-5.6-sol@xhigh。
- §5.1 还提到 **MiniMax sparse attention**(prefill+decode dispatch family)已在 Cake IR 上跑通——sparse attention 家族可行性对 DSA 目标直接相关;KDA prefill 2.05×、decode 1.14×,FlashInfer PR #4262/#4279/#4274/#4287 四个上游 PR。

[[cake-paper.md]] [[tirx-cake-ir-feasibility.md]] [[tirx-harness-gap.md]] [[tirx-wgmma-gap.md]]
