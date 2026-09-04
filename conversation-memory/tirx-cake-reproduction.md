# 基于 tirx 复现 CAKE 效果的规划

- **主题**:项目目标(在 tirx 上按 CAKE 方法魔改复现部分效果)、当前决策与排序
- **时间**:2026-08-28
- **来源**:本轮对话结论;见 [[cake-paper.md]]、[[tirx-wgmma-gap.md]]

## 目标与当前决策

- **目标**:魔改 tirx 编译栈,按 CAKE 论文的方法复现部分效果。
- **硬件**:H20 / H200(注意:两者都是 **sm_90a Hopper**,不是 Blackwell)。
- **首选 workload**:KDA(Kimi Delta Attention)prefill。

## 关键约束(本轮定下来的、要避免踩的坑)

1. **H20 vs H200 差很远**:H20 是低算力/高带宽的特供阉割片,KDA prefill 是 GEMM+投影的 compute-bound 负载,H20 会被算力拖死、信号脏 → **只在 H200 上做**。
2. **KDA prefill 是前沿硬核 workload**:线性注意力 + 跨 chunk 保持活跃的 recurrent state(delta rule)+ fixed/packed-variable/tail 三类输入;tirx 及本地 **没有任何现有 attention/delta kernel 可改**(`tirx-kernels` 也不在本地)。
3. **三个硬问题严格排序**,不能一起开工:
   - ① substrate 缺口:Hopper 快路径缺 WGMMA(见 [[tirx-wgmma-gap.md]]);
   - ② KDA schedule 要手写(仓库无先例,且论文 2.05× 是 B200 测的,Hopper 无直接可比基线);
   - ③ agent harness 整个不存在(候选生成/rank/evaluate/route、cost model、本地化诊断)。

## 决策排序(substrate 先达标,不是先上 agent)

1. 先补 sm_90a WGMMA 路径(这是 CAKE 论文外层循环"补能力缺口"的教科书案例,本身就是 compiler-evolution 那一章的素材);
2. 手写正确 KDA prefill schedule,测与 FlashKDA/Hopper 基线的差距;
3. substrate 达标后,再搭最小 agent harness 做 clean-start 对照。

## 复现结论分层

- **能复现**:方法论架构、Table 2 式可归因对照实验(tirx-IR 臂 vs 直接 CUDA 臂)、compiler-evolution 外层、加上"failure→verifier rule"的内循环。
- **难复现**:数字幅度——依赖 Blackwell 硬件 + GPT-5.6-sol@xhigh 的 80M token 预算 + 数小时进化时间 + 大语料 + 已达标 codegen。

## 下一去险切片

不追求通用 layout,先锁**单 shape M×N×K(BF16)+ 单 layout + 寄存器累加器**,与 CUTLASS 数值对拍,拿到第一个 bit-accurate WGMMA,再谈泛化 + 调优。

## 后续进展(2026-09-04)

- Cake IR 逐构造复现可行性评估(多数构造 ✅,layout 封装层 / WGMMA / cost model 三卡点)见 [[tirx-cake-ir-feasibility.md]];
- harness 卡点分析(空壳同步 verifier、语料基质、cost model 设计空间与降级路径)见 [[tirx-harness-gap.md]]。