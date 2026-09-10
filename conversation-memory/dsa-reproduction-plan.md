# DSA 在 tirx 上的逐步复现计划(主计划)

- **主题**:workload 定为 DSA(DeepSeek sparse attention),在 H200(sm_90a)上逐步复现 CAKE 效果;分层推进、decode 为冲刺目标
- **时间**:2026-09-09(环境侦察 + 方向决策),对应此前 [[tirx-cake-reproduction.md]](KDA 计划,已被本计划取代 workload)
- **来源**:本机侦察(nvidia-smi/torch/pynvml/git/code 复核)+ 用户决策;依据 [[tirx-wgmma-gap.md]]、[[tirx-harness-gap.md]]、[[tirx-cake-ir-feasibility.md]]、[[cake-paper-deep-notes.md]]

## 决策记录(2026-09-09)

1. **workload: KDA → DSA**(用户定)。理由:论文 Table 4 里 DSA 相关行信号最强——S5 lightning indexer 是 CAKE 最大胜场(FP8 1.270×/FP4 1.273×),S8 DSv4 sparse MLA decode 是**唯一失败行**(0.9649);且本机已装 flashinfer 0.4.1(decode 基线现成);DSA 是生产级热点(sparse attention 家族在 Cake IR 已跑通 MiniMax 先例)。KDA 侧无 H200 可比基线且 2.05× 为 B200 数字,优先级让位。
2. **主攻方向:分层推进,sparse decode 为冲刺目标**(用户定)。骨架不变:substrate → DSA 组件 → agent harness。
3. **节奏:每阶段数值 gate 停下汇报 + 更新 conversation-memory**(用户定)。
4. 沿用既定约束:论文绝对数字是 B200+GPT-5.6-sol@xhigh+80M token 产物,**只复现方法相对形态 + H200 本地基线达标**;语料是产物不是前提;cost model 初期以 GPU 实测顶替。

## 环境事实(2026-09-09 侦察,写入时以此为准)

| 项 | 事实 |
| --- | --- |
| GPU | 名 "NVIDIA L20X"(掩码),**实测 sm_90a**:torch/pynvml 均 capability (9,0),132 SM,139.8 GiB 可用,PCIe,driver 570.148.08,CUDA 12.8 → 判定 H200 级 GH100。**注意 nvidia-smi `compute_cap` 列报 8.9 是掩码假象,一律以 torch/pynvml 为准** |
| 工具链 | python 3.10.13(/opt/conda)、gcc 10.2.1、cmake 3.26.4、ninja 1.11.1、**无 ccache** |
| 网络 | pypi / github 可达 |
| 预装 | torch 2.8.0+cu128、flashinfer-python 0.4.1、triton 3.4.0、atorch 1.6.3、flash-linear-attention 0.3.2、nvidia-cutlass-dsl 4.2.1、torchao;无 deepgemm / flashmla(可 pip) |
| 仓库 | /root/code/tvm **无 build/**,git submodule 未 init,**无 tirx-kernels** workspace;HEAD=78b73c569(仅 cake.pdf + conversation-memory) |
| tirx WGMMA 现状(复核) | `op.py:705` 仅 encode_matrix_descriptor/noop_barrier intrinsic,**无 mma_async 本体**;`script.py:105-138` CudaWgmmaNamespace 在;`ptx/table.py:7140` 六组 wgmma.mma_async 已注册(cert_arch sm_90a)、`:2438 _wgmma_acc_lanes(m)` 在;`cpp/descriptors.py:39` 描述子编码在;`target_tags.py:60` 只有 nvidia-h100→sm_90a(无 h200 标签);`gemm_async/` 仅 tcgen05.py、`gemm/` 仅 mma_m16n8k_.py → **缺 = intrinsic + dispatch 变体,零件齐备** |

## DSA workload 分解(以论文 Table 4 S5/S8 为锚)

| 组件 | 语义 | 论文对照 | H200 基线(候选) |
| --- | --- | --- | --- |
| A. lightning indexer(=MQA indexer) | 每 query×head group 算 page 级 score(q·k 块归约)→ top-k page 选择;FP8、page 表、ragged | S5:FP8 1.270×/FP4 1.273×(CAKE 最大胜场) | DeepGEMM(pip,Hopper 原生) |
| B. dense MLA decode(中间门) | page-based KV + FP8 group 反量化 + qk→softmax→pv | S7 CUTLASS MLA decode 1.2174/1.1297(DSv3,B200) | flashinfer 0.4.1 MLA decode / FlashMLA(pip) |
| C. sparse MLA decode(=DSA decode) | B + indexer 输出 page 子集;ragged query、H=128、Dqk=Dv=512、page 64/128 | S8:0.9649,**论文唯一失败行**(参考 8942+9125 LOC vs Cake 1393) | 官方 H200 基线存在性待 P0-0.4 recon |

## 成功标准(三档,gate 依次过)

- **T1(substrate)**:tirx wgmma BF16 GEMM 单→多 shape 数值 bit-accurate vs CUTLASS/torch;`gemm_async→wgmma` dispatch 变体落地。
- **T2(组件 parity)**:dense MLA decode、indexer 在 DSA 形状集上达 H200 官方/社区基线 ~90%+ 或 ≤10% 差距;sparse decode 数值正确可跑。
- **T3(decode 冲刺 + 方法论)**:sparse MLA decode 达 H200 现有最好水平或以上;**任一组件跑通 clean-start A/B**(tirx-IR 臂 vs 直写 CUDA 臂),CAKE 方法论形态复现。

## Phase 0 — 机器就绪与基线侦察(0.5–1 周,gate:构建+冒烟过、形状集与基线表落盘)

- **0.1 构建**:`git submodule update --init --recursive` → cmake(USE_CUDA=ON,按 AGENTS.md 约定,复用 build/)→ ninja 构建 → `import tvm.tirx` 冒烟。首建无 ccache,预计 40–90 min。
- **0.2 tirx H200 测试面摸底**:现有测试入口需 Blackwell + tirx-kernels(记忆:本地无)→ 建 **H200 可跑子集 smoke list**(标哪些 pytest 在 sm_90 能过);clone tirx-kernels(网络可达)。
- **0.3 DSA 形状集规格化**:锚 S8(Sq ragged [3,4,5]、H=128、Dqk=Dv=512、page 64/128)+ S5,扩 H200 生产形状(B、top-k、FP8 dequant group 等)→ 落 shapes.md。
- **0.4 基线 survey + microbench**:确认 deepgemm / flashmla / flashinfer 0.4.1 MLA decode / DeepSeek 官方 DSA 内核在 H200 可用性;记录各组件目标形状数字(black-box,只看 API)。**若 sparse decode 官方 H200 基线不存在 → 记录回退基准策略**(组件对拍全覆盖 + 自建直写 CUDA 臂留 P3.3)。
- **0.5 参考纪律固化**:PyTorch 级语义(DeepSeek 开源 modeling 代码)= correctness oracle;官方内核 = black-box timing;**禁止读底层 kernel 实现**。
- 产出:`shapes.md` + `baselines.md`;P0 结束时本计划附档更新。

## Phase 1 — WGMMA substrate(1–2 周;gate = T1)

- **1.1 去险切片**(记忆既定):单 shape M×N×K BF16、单 layout、寄存器累加 → 新 intrinsic `cuda_wgmma_mma_async`(op.py 照 call_intrin 模式,渲染走 ptx 表,可挂 `_wgmma_acc_lanes`)+ 新 dispatch 变体 `gemm_async→wgmma`(`gemm_async/wgmma.py`,对标 tcgen05.py 但砍 TMEM/FP8/SF,累加器留寄存器)。数值 vs CUTLASS/torch,拿**第一个 bit-accurate**。
- **1.2 泛化**:多 shape + TMA 双缓冲 + pipeline stages=2–3(复用 pipeline.py full/empty 配对,验 Hopper 语义);FP8 wgmma(TMA store 的 swizzle 与描述子对齐——**错是静默算错不是 crash**,必做数值对拍);cluster 可选。
- **1.3 target 标签**:补 `nvidia-h200→sm_90a`(当前仅 h100;不补则先用 nvidia-h100)。
- **1.4 语料**:首个验证通过的 WGMMA kernel 入语料台账(第 1 条)。
- 风险:累加器 C-fragment lane 映射(warpgroup 128 线程/64 lane×2 warp);无 kernel 级调试器 → 数值 diff 定位。

## Phase 2 — DSA 组件手写(2–4 周;gate = T2 各子项,每子项:数值 gate → 入语料 → perf 迭代)

- **2.1 dense MLA decode(先建通路)**:page-based KV + TMA 加载 + FP8 group dequant + wgmma qk/pv + softmax;先规整 B 过门再 ragged。对拍 flashinfer 0.4.1 MLA decode / flashmla。→ 语料+1
- **2.2 lightning indexer**:page score GEMM + 跨 page top-k;FP8;对拍 deepgemm indexer。→ 语料+1
- **2.3 sparse MLA decode**:2.1+2.2 输出 page 子集 + ragged query;数值 vs PyTorch oracle;perf 对官方/社区基线(P0-0.4 定)。**论文唯一失败行,研究冲刺点**。→ 语料+1
- 全程:每个修掉的同步/内存 bug 记入 **verifier 规则素材清单**(P3.2 的原料),严格参考纪律。

## Phase 3 — 最小 agent harness + 方法论 A/B(3–6 周,尾段与 P2 并行;gate = T3)

- **3.1 最小闭环**:候选(schedule 脚本)→ parse-time 静态 verify(现成 CRTP verifier 链)→ 数值对拍 → GPU bench(cost model 以实测顶替,降级路径既定)。先人工驱动再 agent 驱动。
- **3.2 同步语义 verifier 从空壳长出**:`AsyncStructsVerifier`/`DeviceFuncVerifier`(verify_tirx_well_formed.cc:146-176,现仅拒 SBlock)→ 补 mbarrier arrive/wait 配对、phase 奇偶、full/empty 配对、race 粗查(pre-compile 挡 hazard;运行时归因只做粗分类)。素材 = P2 累积 + 语料 gate。
- **3.3 clean-start A/B**:选定组件(建议 indexer 或 dense decode 先行)跑 tirx-IR 臂 vs 直写 CUDA 臂,复现 Table 2 形态。
- **3.4 stretch**:layout 降负封装层(offset+swizzle tag→TileLayout,可行性路 A,见 [[tirx-cake-ir-feasibility.md]])、cost model v1(roofline+瓶颈四分类,H200 自校准)、route/dispatcher 多 shape 外环。

## 风险与回退

1. 硬件身份:掩码名 L20X,但 cc 9.0 / 132 SM / 140 GiB 三证据一致 → H200 级,风险低;sm_90a 不变则计划不变。
2. nvidia-smi compute_cap 列 8.9 ≠ 实测 9.0 → 所有脚本以 torch/pynvml 判 cc。
3. sparse decode 官方 H200 基线可能不存在 → 回退:组件对拍(T2 全覆盖)+ 自建直写 CUDA 臂(P3.3);或降级为"数值正确 + 相对自身加速 + vs dense 上界比较"。
4. tirx 测试需 Blackwell → 建 H200 子集;语料 gate 自定义(每条 kernel:数值对拍 + 入台账)。
5. token/模型预算 ≠ 论文 → 只复现相对形态(既定)。
6. target 标签无 h200 → P1.3 补或用 nvidia-h100。
7. 首建无 ccache → P0 一次到位,后续增量构建。

## 进展日志

### P0(2026-09-09,当日完成,gate 通过)

- **P0.1 构建 ✅**:submodule init → cmake(USE_CUDA=ON)→ ninja(192 核 ~10min)→ `import tvm.tirx` 冒烟过(tvm 0.26.dev0)。**注意**:`tvm_ffi` 必须用 submodule 版(PYTHONPATH 首放 `3rdparty/tvm-ffi/python`),site-packages 里是旧版会 import 失败。
- **P0.2 测试面 ✅**:2709 项可收集;**H200(cc9.0)上 tile_primitive CUDA 套件被 `has_cuda_compute(10)` 全 skip**;可跑且过 = 编译器侧 96 + operator 143 + codegen 154;**codegen 9 failed 全 PTX-dialect/tensor-map 侧,源码零改动 → 预存在,记 [待查] 不入关键路径**。
- **P0.3 形状集 ✅**:[[shapes.md]]。
- **P0.4 基线 ✅(部分降级)**:[[baselines.md]]。indexer(deep_gemm,sm90)数字落地;**dense MLA decode 四条候选基线全断**(flashinfer SM80-only、cutlass_mla_decode SM100-only、trtllm-gen 需拉 NVIDIA cubin、FlashMLA 被 gcc10.2 段错误卡死)→ 走回退(PyTorch oracle + 自建对拍);GEMM 基线 cuBLAS BF16 810 TFLOPS 落地供 P1。
- **gate 结论:P0 通过,可进 P1(WGMMA substrate)**。dense decode 基线阻塞不挡 P1。

## 台账指针

- 语料台账:`conversation-memory/corpus.md`(初始为空;每验证通过 kernel +1,记 shape/数值/parity 来源)。
- 进度:本文件随 gate 追加"进展日志"小节(日期 + 结果 + 偏差)。
- 相关:[[tirx-wgmma-gap.md]] [[tirx-cake-reproduction.md]] [[tirx-harness-gap.md]] [[tirx-cake-ir-feasibility.md]] [[cake-paper-deep-notes.md]] [[tirx-concept-isomorphism.md]]
