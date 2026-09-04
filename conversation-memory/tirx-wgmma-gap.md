# tirx 的 Hopper WGMMA 能力缺口

- **主题**:tirx 张量核 gemm 路径现状(只有 mma.sync / tcgen05,缺 wgmma),缺口评估与工作量
- **时间**:2026-08-28
- **来源**:`python/tvm/backend/cuda/tile_primitive/gemm/`、`gemm_async/`、`ptx/table.py`、`cpp/descriptors.py`、`op.py`、`script.py` 的 grep 核对

## 现状(tirx 支持 Hopper 的"正确性",不支持"高性能张量核")

精确表述不再是"完全不支持 Hopper",分三层:
1. **Hopper 是正式 target**:`target_tags.py:59-61` 有 `nvidia/nvidia-h100 → sm_90a`;TMA(`copy_async/tma.py:336-338,769-770` 只 gate `_target_sm >= 90`,仅 gather4/cta_group=2 要 SM100)、reduction、elementwise、mbarrier `pipeline.py`(全文无 arch 检查,docstring 的 "SM100" 只是惯例措辞)都不依赖 Blackwell,H200 可用。
2. **张量核 gemm 只有两条,都不是 Hopper 快路径**(2026-09-04 代码复核确认:`gemm/` 目录仅 `mma_m16n8k_.py`,`gemm_async/` 仅 `tcgen05.py`):
   - 同步 `gemm` 唯一变体 = `"mma.m16n8k*"`(`mma_m16n8k_.py`,Ampere 老路径 mma.sync,Hopper 向后兼容但慢);
   - 异步 `gemm_async` 唯一变体 = `"tcgen05"`(`tcgen05.py`,Blackwell 专属 SM100,H200 上跑不了)。硬锁机制:累加器强制 TMEM(`tcgen05.py:403-414` 校验 `C_scope=tmem`)+ PTX 表条目 `cert_arch="sm_100a"`(ptxas 在 sm_90a 直接拒编)。
3. **WGMMA 缺的就是一个 `gemm_async` 变体**,不是缺全部。

## 已预埋的 WGMMA 基础设施(这是本缺口"比 naive 估计小很多"的原因)

- `ptx/table.py`:全套 `wgmma.mma_async`(f16/bf16/tf32/fp8/int/b1,`:7140-7177` 起六个 type group,均 `cert_arch="sm_90a"`)+ **累加器 fragment lane 映射 `_wgmma_acc_lanes(m)`**(`:2438-2440`)+ `wgmma.fence/commit_group`(`:7420-7432`)、`wgmma_wait_group`(`:7451-7462`)同步族;
- **`backend/cuda/cpp/descriptors.py:36-41`**(注意:不是 `tile_primitive/cpp/`,此前记录路径有误)有 `cuda_wgmma_encode_matrix_descriptor`(纯 C bitfield 填充);`op.py:705-706` 有对应 intrinsic;
- `script.py:101-106,138` 已有 `T.cuda.wgmma` 命名空间(带 `noop_barrier` + `encode_matrix_descriptor`)。零件齐备但**无任何 dispatch 变体消费**。

推断:当初写 tcgen05 时按 WGMMA 把通路预埋了。

## 要新写的

1. 一个 intrinsic `cuda_wgmma_mma_async`(op.py 现只有 descriptor + fence,无 mma_async 本体),照 tcgen05 的 `call_intrin` 模式补,渲染交 ptx 表;
2. 一个 dispatch 变体 `gemm_async → wgmma`(对标 `tcgen05.py` 1527 行,但 BF16-only 更小):累加器留寄存器(无 TMEM 分配/ldst,砍掉 tcgen05 最大复杂度)、砍掉 block-scaled FP8/SF。

## 工作量估计(人周,前提:已熟 tirx 布局/分派机制)

| 目标 | 估算 |
| --- | --- |
| 单 shape / 单 layout / BF16 **正确** WGMMA | ~1–2 人周 |
| 近 CUTLASS 多 shape / 多 pipeline 深度 | +2–4 人周 |
| 足以当 KDA prefill substrate | 总计 4–7 人周 |

## 难度集中在哪(调试而非代码量)

1. 累加器 **fragment 布局**(warpgroup 级 128 线程 / 64 lane×2 warp,C-fragment TileLayout 映射到 lane 分布);
2. **swizzle / 描述子对齐**(B 操作数 wgmma SMEM 描述子 swizzle mode 必须与 TMA store 对齐,错了是静默算错,不是 crash)——正是 Cake 说的"layout 代数让错误难定位"一类;
3. 异步 pipeline 编排(`.mbarrier::complete_tx` + `wgmma.wait_group`,复用 `pipeline.py` 需确认 Hopper 语义下退得下来);
4. Hopper 无 kernel 级调试器,只能数值 diff 定位。

详见 [[tirx-cake-reproduction.md]] 的"去险切片"。