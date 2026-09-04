# tirx 的 layout 代数 —— 与 CAKE「无 layout 代数」相反

- **主题**:tirx 有一套一等公民的 layout 代数,与 Cake IR「不做 layout 代数」的设计取向相反
- **时间**:2026-09-01
- **来源**:`include/tvm/tirx/layout.h`、`src/tirx/ir/layout/*.cc`、`python/tvm/tirx/layout.py`、`python/tvm/backend/cuda/tile_primitive/{tma_utils,layout_utils}.py`、`python/tvm/backend/cuda/lang/smem_desc.py` 核对

## 结论

CAKE IR 的卖点是「**故意不做 layout 代数**」——agent 直接写具体存储/访问决策(SMEM view offset、字节偏移、TMEM 列范围、swizzle tag),编译器查生产/消费兼容,从而避开 CuTe/Triton layout algebra 学习负担。

tirx 押的是**另一边**:layout 是一等公民、完全可组合的代数系统。这是对上次 [[cake-paper.md]] 里「概念 1:1 同构」的修正——唯独「无 layout 代数」这条不成立,方向相反。

## 证据链

### 1. 一等 layout 类型 + 全套代数运算(`include/tvm/tirx/layout.h`)

- `Layout` 抽象基类(line 52–134),运算:`Tile` / `DirectSum` / `Slice` / `Canonicalize` / `IsTileInner` / `IsTileOuter` / `IsDirectSumLeft` / `IsDirectSumRight` / `CompatibleWithShape` / `VerifyWellFormed` —— 这就是「合成 + 分解 + 判断」的完整代数。
- `TileLayout`(line 311–409):`shard` + `replica` + `offset`,`Iter(extent, stride, axis)` 区分 thread axis / memory axis —— CuTe 式 `shape:stride` 平铺布局。
- `ComposeLayout`(line 411–495):`per_element / swizzle_len / atom_len / swizzle_inner` —— 显式 swizzle 布局。

### 2. Python 侧是可手写的 DSL(`python/tvm/tirx/layout.py`)

- `S[shape:stride] + R[shape:stride] + offset` 组合语法、`3 @ Axis.laneid` 轴绑定(line 549 `__rmatmul__`、1213–1300 `_SpecBuilder`)。
- `canonicalize()/tile()/direct_sum()/unpack/pack/broadcast/permute_dims/to_psum/storage()` 等全部暴露。
- 一群照 PTX ISA 表手写的 fragment 布局工厂:`tmem_datapath_layout`(622)、`tmem_mma_operand_layout`(764)、`tcgen05_atom_layout`(915)、`wg_local_layout`(876),逐 lane 算 `wid_in_wg/laneid` bit 分解。

### 3. swizzle 是显式暴露给 kernel 编写者的概念

- `src/tirx/ir/layout/compose_layout.cc::ApplyStructured`:几百行「carry-free」整数证明,推导 swizzle 地址 `phase = (B/atom)&mask; addr = B + (D ^ (phase<<per_element)) ^ K`。
- `tma_utils.py:37` `mma_atom_layout(dtype, swizzle_mode) -> ComposeLayout` 把 `SwizzleMode` 换算成 `swizzle_len/atom_len`。
- `lang/smem_desc.py` `SmemDescriptor.init(smem_ptr, ldo, sdo, swizzle)` 把 `swizzle` 作为用户参数。

这套对应的是 CuTe `Layout/Composition/LogicalProduct/Swizzle` + Triton 隐式布局,正是 Cake 说它刻意省掉的东西。

## Nuance(拆开「考虑 layout 计算」两解)

- **解读 A:「IR 是否表达 layout」** → tirx 明确反对(有一等 layout 类型),和 Cake 相反。
- **解读 B:「agent 要不要亲自做布局数学,还是编译器自动推导 + 校验」** → tirx **部分**靠近 Cake:
  - 用高层 tile primitive / allocator 时,layout 由 `tmem_datapath_layout` 等工厂**从 PTX ISA 表自动推导**,`tmem_pool.alloc(..., layout=...)` 埋进 buffer,dispatch 用 `Canonicalize/IsTileInner` 做结构等价校验(`layout.py:602` 注释原话:「让 dispatch structured verify atom↔datapath 兼容性,而非默默接受错配」)。
  - **但**校验机制本身仍是那套 layout 代数,且 `S/R/ComposeLayout` 从 IR 表面到 kernel 编写全程可见。写 tirx kernel 的 agent 仍要直面 CuTe 式布局推理(swizzle 参数、canonical form、tile/direct-sum 分解),没有 Cake 那种「只写字节偏移 + swizzle tag、让编译器查兼容」的降负。

## 对复现的影响

- 上次 `cake-paper.md` 第 2 点「无 layout 代数 ↔ tirx tile_primitive/DispatchContext」映射不成立:`DispatchContext` 解决「按 target/tile 形态选算子」,不是 Cake 去 layout 代数的对应物。
- 复现 CAKE「agent 编辑无 layout 代数的类型化 IR」卖点在 tirx 上**不是免费的**:tirx 的核写面是 CuTe 型,要降负需额外加一层「具体 offset + swizzle tag 决策、编译查兼容」的封装。

[[cake-paper.md]] [[tirx-cake-reproduction.md]] [[tirx-wgmma-gap.md]]