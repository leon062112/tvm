# tirx —— TVM 新一代张量级 IR

- **主题**:tirx 的定位、与原生 tir 的关系、子系统组成与设计意图
- **时间**:2026-08-28
- **来源**:`python/tvm/tirx/__init__.py`、`include/tvm/tirx/`、`src/tirx/`、`python/tvm/tirx/compilation_pipeline.py`、`.agents/skills/tir-test/SKILL.md`

## 一句话定位

`tvm.tirx` 是 TVM **Tensor-level IR 的新一代命名空间**(`__init__.py`: "Namespace for Tensor-level IR"),与原生 `tvm.tir` **并列共存**、复用 TVM 编译器底座(runtime、IRModule、op 注册、`tvm.script`、FFI),专门面向现代 GPU(尤其 Blackwell,compute capability ≥ 10)的 kernel 生成,配合外部 `tirx-kernels` 仓库使用。

## 关键定性

- **不是"构建在 TVM 之上"的独立项目**,而是同一仓库内、与原生 TIR 同级的一套新 IR + 编译栈。
- **是嵌入式 DSL(EDSL),非独立语言**:通过 TVMScript 的 `"tirx"` dialect(`@T.prim_func` + `tvm.script.register_dialect("tirx", "tvm.tirx.script")`)和 `lang/` Python 高层抽象表达。
- **与 `s_tir` 协作**:tirx 提供 IR / 算子 / 语言;s_tir 提供调度 / 自动调参(meta_schedule、dlight、tensor_intrin),共享 `SBlock`。

## C++ 侧

`include/tvm/tirx/`:`expr.h` `stmt.h` `var.h` `buffer.h` `function.h` `op.h` `builtin.h` `op_attr_types.h` `layout.h` `index_map.h` `exec_scope.h` `exec_context.h` `tile_primitive.h` `expr_functor.h` `stmt_functor.h` `transform.h` `analysis.h` `script/`

`src/tirx/`:`ir/`(含 `layout/`、`script/`)、`op/`、`analysis/`、`transform/`、`script/`(builder + printer)

## Python 侧子模块(`python/tvm/tirx/`)

- **IR 核心**:PrimFunc、Buffer、Var、For、BufferLoad/Store、IndexMap、全套算子;**新增** `SBlock`/`SBlockRealize`(取代 Block)、`ExecScope`/`ScopeIdDef`、`TileLayout`/`ComposeLayout`、`ScopeIdDefStmt`;`LetStmt` 已折叠进 `Bind`。
- **`lang/`** —— GPU kernel 语言层:`alloc_pool`、`smem_desc`、`warp_role`、`tile_scheduler`、`pipeline`。
- **`operator/tile_primitive/`** —— 瓦片原语注册表 + 分派器(`common`/`ops`/`registry`/`dispatcher`);顶层 `TilePrimitiveCall` / `DispatchContext` / `LambdaExpr`。
- **`analysis/`** —— verify_well_formed / ssa / memory、var_use_def、side_effect、deep_equal、expr_complexity、exec_context、stmt_finding。
- **`transform/`** —— Lowering pass:`LowerTirx`(及 `_cleanup/_dedup_tensormap/_opaque`)、`FlattenBuffer`、`VectorizeLoop`、`UnrollLoop`、`StmtSimplify`、`StorageRewrite`、`NarrowDataType`、`LowerIntrin`、`MakePackedAPI`、`SplitHostDevice`、`RemapThreadAxis`、`UpdatePointerStorageScope`、`RemoveAssume`、`DTypeConversion`、`BF16/FP8ComputeLegalize` 等。
- **`build.py` + `compilation_pipeline.py`** —— TIR 后端编译入口:`build()`、`get_tir_pipeline` / `get_default_tir_pipeline` / `register_tir_pipeline`;**纯运行时库(`TVM_USE_RUNTIME_LIB=1`)下不导入**。
- **`script/`** —— TVMScript tirx dialect 的 `parser/` + `builder/`(含 `triton.py`、`external_kernel.py` 互操作)。

## 设计意图

1. 面向现代 GPU(Blackwell/CUDA 10+)的瓦片计算原语(`TileLayout`、`TilePrimitive` 分派、`smem_desc`、`warp_role`、软件流水线)。
2. 更结构化的执行域(`ExecScope` / `ScopeIdDef`)。
3. 可分派算子(`DispatchContext` + 注册表,按 target / tile 形态动态分派)。
4. 命名空间隔离,允许在不破坏原生 TIR 兼容性的前提下做较大 IR 重构。

## 测试 / 运行

入口:`.agents/skills/tir-test/SKILL.md`。需 Blackwell GPU + `tirx-kernels` 仓库,`xdist` 并行;环境:`PYTHONPATH="<ws>/tirx-kernels:<ws>/tvm/python"`、`TVM_LIBRARY_PATH="<ws>/tvm/build/lib"`,按显存占用选 GPU。
