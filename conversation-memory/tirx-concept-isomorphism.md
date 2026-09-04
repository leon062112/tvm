# tirx ↔ CAKE 概念同构的代码级证据

- **主题**:CAKE 四大核心概念 + 执行域底座在 tirx 中的命名级对应(file:line 证据,2026-09-04 代码复核)
- **时间**:2026-09-04
- **来源**:并行代码探查;`python/tvm/backend/cuda/lang/{smem_desc,alloc_pool,warp_role,pipeline,tile_scheduler}.py`、`python/tvm/tirx/operator/tile_primitive/dispatcher.py`、`include/tvm/tirx/{exec_scope,exec_context,tile_primitive}.h`

## 证据表

| CAKE 概念 | tirx 对应 | 关键 file:line |
| --- | --- | --- |
| 声明式 smem/tmem 资源 | `SmemDescriptor` / `SMEMPool` / `TMEMPool` | `smem_desc.py:24-42`(@T.meta_class,init 编码一次、add_16B_offset 复用);`alloc_pool.py`:SMEMPool L395/422-430(bump allocator,alloc(shape,dtype,scope,align,layout))、TMEMPool L188(alloc L262-292 直接 decl_buffer scope="tmem" + allocated_addr + layout;alloc_tcgen05_mma_A/D L324-344) |
| 显式 warp 角色 | `WarpRole` / `WarpgroupRole` | `warp_role.py:41-89`:角色 = warp_id guard + setmaxnreg 寄存器预算 + 代码块三者绑定(`__enter__` L75-82 生成 if + setmaxnreg);文件头 L22-35 示例即 tma_warp(regs=48)/mma_warp(regs=232) 分工;`WarpgroupRole` L92-150 支持 wg 区间 |
| pipeline+barrier 生产-消费 handoff | `MBarrier`/`TMABar`/`TCGen05Bar`/`Pipeline`/`PipelineState` | `pipeline.py`:MBarrier L142(init/wait/arrive)、TMABar L252(arrive.expect_tx)、TCGen05Bar L289(tcgen05.commit)、Pipeline L330-375("a full/empty mbarrier pair",_BAR_KINDS 按 tma/tcgen05/mbar 配对)、PipelineState L46-82(stage/phase 相位环);实际用例 `tile_scheduler.py:872-944`(full.wait 消费方等待、empty.arrive remote=0 跨 CTA 释放走 cluster mapa) |
| 可分派算子(按 target/tile 形态) | `_DISPATCH_TABLE` + `register_dispatch` + `DispatchContext` | `dispatcher.py:82-136`(DispatchCase: variant/priority/preds/impl,按 (Op, target_kind) 键控)、run_dispatch L239-301(按优先级排序、fail(reason) 试下一变体);`tirx/tile_primitive.py:252-318`(TilePrimitiveCall 六字段 op/args/workspace/config/dispatch/scope + __init_subclass__ 自动注册)、DispatchContext L62-243(target/exec_scope/launch_params/var_range_map + is_cuda/is_warp/is_cta 谓词);注册示例 `gemm/mma_m16n8k_.py:260-270`、`gemm_async/tcgen05.py:1511-1527` |
| 执行域底座(五级格) | `ExecScope`/`ScopeIdDef`/`ActiveSet` | `exec_scope.h:45-51`(ScopeKind: kCluster=2..kThread=6)、L79-90(ScopeBinding 闭合 parent→cur 枚举)、L99-117(ScopeIdDef,extents NullOpt=deferred 由 verifier BFS 推导);`exec_context.h:58-80`(ActiveSet{TileLayout layout},轴 laneid/warpid/cta_id/wid_in_wg);`tile_primitive.h:188-233`(每个 TilePrimitiveCall 内嵌 scope) |

## 判断

同构非巧合:CAKE 核心作者即 TVM/TensorIR 原班人马(见 [[cake-paper.md]]),tirx 与 Cake IR 是同一批人在两个仓库的平行实现。复现 Cake IR 时前四行是"改名级"工作量;**唯一方向相反的是 layout 哲学**(见 [[tirx-layout-algebra.md]])。复现可行性总评估见 [[tirx-cake-ir-feasibility.md]]。
