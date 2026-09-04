# CAKE harness 在 tirx 上的卡点分析

- **主题**:harness 七组件拆解评级;cost model 设计空间;三个深卡点(空壳同步 verifier / 语料基质 / DSA oracle)
- **时间**:2026-09-04
- **来源**:`src/tirx/analysis/verify_tirx_well_formed.cc` 取证 + 论文 §3–§4/App.B.5/App.C 散点 + 本轮对话综合;上游 [[tirx-cake-ir-feasibility.md]]、[[cake-paper-deep-notes.md]]

## 七组件拆解与评级

| # | CAKE harness 组件 | tirx 现状 | 评级 |
| --- | --- | --- | --- |
| 1 | 静态门四类(safety/conformance/data-consistency/schedule-semantics) | Verifier CRTP + AccessPath 框架完备;ExecScope/ScopeId/Layout 三 verifier 有实内容;**AsyncStructsVerifier/DeviceFuncVerifier 是空壳**(verify_tirx_well_formed.cc:146-176,函数体只有 SBlock 语法拒检) | 🔴 深 |
| 2 | cost model(校准 + 瓶颈归因 + 建议) | 整个缺失 | 🟡 可降级 |
| 3 | codegen + GPU runtime | ✅ 现成 | ✅ |
| 4 | 数值 oracle + CUPTI 计时 + L2 flush + profiler | 计时/profiler 是工程量;DSA oracle 要自建 | 🟢 工程 |
| 5 | sanitizer 反馈回路 | compute-sanitizer 可用,但报 SASS 级,**无映射回 tirx IR 的机制** | 🔴 深 |
| 6 | 证据路由 + 结果留存 + 语料回归门(P6) + merge gate | 全要新建;语料基质缺失(见卡点②) | 🔴 结构性 |
| 7 | agent 工作流编排(generate→filter→evaluate→route) | 纯工程 | 🟢 |

## 重大发现:同步语义分析是空白

`AsyncStructsVerifier`(verify_tirx_well_formed.cc:146-160)与 `DeviceFuncVerifier`(L162-176)函数体均只有「SBlock is not allowed in tirx=True mode」两条拒检——**mbarrier arrive/wait 配对、phase 奇偶、full/empty 配对、race/deadlock 分析全部为零**。CAKE Table 1 的 "Program safety" 门(synchronization/ordering hazard)在 tirx 上不存在。好消息:Pipeline/MBarrier 都是显式 IR 调用,语义可分析,长出真分析有信息基础。

## cost model:设计空间已被论文四句散点约束(见 [[cake-paper-deep-notes.md]])

- 结论:经验校准型解析模型(特征 + 每-target 校准系数 + 瓶颈分类),只承担 rank/filter;
- tirx 声明式资源模型恰好把特征写在 IR 表面(buffer scope/extent、Pipeline stage 深度、WarpRole 分工、TilePrimitiveCall 指令形态)→ 特征提取可行;
- **v1 设计**:roofline 式估计(各存储层字节×H200 校准带宽;MMA 指令数×校准吞吐;占用率粗估)+ 瓶颈四分类(compute/bandwidth/sync/occupancy)+ 误预测回写校准。估 2–4 周可用;
- **降级路径**:纯 GPU 实测排序起步(CAKE Table 2 两臂本来都烧 GPU),cost model 作为第一个 harness 进化项长出。

## 三个深卡点(按深度排序)

1. **运行时/同步失败 → IR 归因链断裂**:pre-compile 半边(空壳同步分析,见上)+ post-compile 半边(hang/illegal-access 只有 CUDA error,sanitizer 报 SASS 地址,codegen 不维护 SASS↔IR 映射)。对策:把 hazard 尽量提到 pre-compile 静态分析挡住(mbarrier 相位/配对先行),运行时归因只做粗分类(hang/nan/mismatch);需要时给 codegen 加 lineinfo 贯穿。这是 harness 复现里真正的新研究/工程量,也是 compiler-evolution 内环的主要素材。
2. **test-gated 语料基质缺失**:CAKE 的 harness 进化靠语料回归门(400+ 静态 + 399 GPU);而 tirx 现有测试入口要 Blackwell GPU + 外部 tirx-kernels 仓库(本地不在),H200 上连 tirx 自身回归都跑不全。对策(App.A 的启示):**语料是产物不是前提**——每个验证通过的 kernel 即入语料,新 verifier 规则只对当前语料 gate,从第 1 个 WGMMA 开始积累。
3. **DSA oracle + 评测管道(工程量,但要早做)**:按论文 reference-access 纪律——agent 不许看 DeepGEMM/FlashMLA 低层实现,但 PyTorch 级参考(DeepSeek-V3.2 开源模型代码)可做正确性 oracle,官方 kernel 做黑盒计时基线。约 1–2 周。CUPTI GPU span + L2 flush 标准工程;ncu 报告解析进证据路由是中等工程。

## 非技术约束

论文数字依赖 GPT-5.6-sol@xhigh + 80M token/轮 × 多轮复跑(见 [[cake-paper-deep-notes.md]] 预算量级)。换模型/降预算后能复现的是**方法论相对形态**(tirx-IR 臂 vs 裸 CUDA 臂对照),不是绝对数字——与 [[tirx-cake-reproduction.md]] 的复现分层一致。

## 一句话

cost model 是最显眼但不是最硬的卡点(有明确降级路径);真正深的是「运行时/同步失败到 IR 决策的归因链」和「test-gated 进化依赖的语料基质」——而做这两件事本身就是在复现论文的 compiler-evolution 外环。
