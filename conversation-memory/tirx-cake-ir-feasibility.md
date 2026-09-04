# 用 tirx 复现 Cake IR 的可行性评估

- **主题**:逐构造映射 Cake IR → tirx,三卡点与工作量估计,实施路径排序
- **时间**:2026-09-04
- **来源**:本轮对话综合(cake.pdf 深读 + 代码取证);证据见 [[cake-paper-deep-notes.md]]、[[tirx-concept-isomorphism.md]]、[[tirx-layout-algebra.md]]

## 前提

Cake IR **不开源**(公开物只有论文 + FlashInfer 上生成的 CUDA PR),复现 = 照论文描述在 tirx 上重建,不是 fork。

## 逐构造映射(结论:多数 ✅,两处 ❌)

| Cake IR 构造(论文 Fig.3 / App.B) | tirx 对应物 | 差距 |
| --- | --- | --- |
| `lm.smem` / `pool.view` / `lm.tmem` | SMEMPool/TMEMPool/SmemDescriptor | ✅ 直接对应 |
| `lm.role(warps=[..])` + `with role:` | WarpRole/WarpgroupRole 上下文管理器(带 setmaxnreg) | ✅ 直接对应 |
| `lm.pipeline` / `lm.barrier` / `lm.wait` | Pipeline full/empty、MBarrier/TMABar/PipelineState | ✅ 直接对应 |
| `smem_q.tma_load(coords, stage, barrier)` | copy_async/tma.py(SM90+ gate) | ✅ 直接对应 |
| `lm.mma(tmem_acc, smem_q[stage], ...)` | gemm_async → tcgen05 变体 | ⚠️ sm_100a 通;**H200 缺 WGMMA 变体** |
| 类型化词汇 + op 分类 + 本地化诊断 | TIRxOpCategory + Verifier CRTP + AccessPath 报错 + dispatch fail(reason) | ✅ 机制现成,覆盖面要扩 |
| 确定性 lowering → 可检视 CUDA | tirx lowering pipeline + codegen | ✅ TVM 传统强项 |
| **无 layout 代数**(agent 写字节偏移/swizzle tag) | tirx 是一等 layout 代数,全程可见 | ❌ 哲学相反 |
| **cost model + 瓶颈归因报告** | 无(meta_schedule cost model 是调度搜索用,不是归因报告) | ❌ 整个缺失 |

## 三卡点

1. **layout 哲学(结构性分歧,决定复现的是不是"真 Cake IR")**:
   - 路 A:封装层——agent 表面只写 offset+swizzle tag,封装层翻成 TileLayout(工厂函数 mma_atom_layout/tmem_datapath_layout 已把 PTX 规范编码好)+ canonicalize/assert_structural_equal 查兼容(机制现成,如 tcgen05.py:341-351)。估 2–4 周。**走这条路才算复现 Cake IR**。
   - 路 B:放弃该卖点,agent 直面 layout 代数——复现其余 90% 但丢掉论文最核心的降负目标。
2. **Hopper WGMMA(substrate 缺口,仅在坚持 H200 时是卡点)**:CAKE Table 3 的 sm_90a 路径也是 WGMMA(见 [[cake-paper-deep-notes.md]]);tirx 零件全预埋(见 [[tirx-wgmma-gap.md]]),缺 gemm_async→wgmma 变体。注意:**若复现目标改 Blackwell,此卡点消失——H200 限制与 Cake IR 复现是两件事,别绑死**。
3. **cost model(全新组件)**:CAKE 用它 rank/filter 候选 + 给瓶颈归因;tirx 无对应物。但它是**效率件不是正确性门槛**(CAKE 自己 GPU 实测为 ground truth),第一版可纯实测顶替。设计空间见 [[tirx-harness-gap.md]]。

## 实施路径(按依赖排序)

1. Cake 风格表面层(1–2 周):在 tirx lang/ 上包 schedule()/role/pipeline/barrier 声明式 API——@T.meta_class 机制已被 SmemDescriptor/Pipeline 证明可行;
2. layout 降负封装(2–4 周,走路 A):offset/swizzle tag → TileLayout 翻译器 + 兼容性 verifier 规则挂进现有 Verifier 链;
3. WGMMA 变体(若留 H200;正确版 1–2 周)——与 1、2 并行;
4. 最小 agent 闭环:parse-time verifier + 数值对拍 + benchmark,纯实测顶替 cost model;
5. cost model、failure→verifier rule 闭环、tactic 库:作为外层进化逐步长出(本身就是复现论文方法)。

## 判断

**可行,且比任何其他底座(Triton/CuTe/裸 CUDA)都可行**——概念同构是命名级的;真正要新建的只有 layout 封装层、(H200 场景)WGMMA 变体、cost model 三件,前两件的零件都已埋在代码里。

[[tirx-cake-reproduction.md]] [[tirx-harness-gap.md]] [[tirx-concept-isomorphism.md]]
