# Apache TVM 项目概览

- **主题**:TVM 的定位、目录结构、关键模块与开发约定
- **时间**:2026-08-28
- **来源**:`README.md`、`AGENTS.md`、仓库根目录结构、git log

## 定位

Apache TVM(Apache 软件基金会顶级项目)是**开源机器学习编译器栈**:把深度学习模型从高层框架(ONNX / PyTorch / TF 等)编译为可在异构硬件(CPU / GPU / NPU / 加速器)上部署的模块。

核心理念:Python 优先(快速自定义编译流水线)、通用部署(最小可部署模块)。

## 目录结构

| 目录 | 作用 |
| ---- | ---- |
| `include/tvm/` | C++ 公共头文件 |
| `src/` | C++ 实现(编译器 + 运行时) |
| `python/tvm/` | Python 包(绑定 + 高层 API) |
| `tests/` | C++ / Python / 集成 / lint 测试 |
| `cmake/` | CMake 模块与默认配置 |
| `3rdparty/` | 第三方依赖与子模块 |
| `docs/` | 文档源码 |
| `apps/` | 应用示例 |
| `jvm/`、`web/` | JVM、Web 端集成 |

## 主要 IR / 模块(`src/` 与 `python/tvm/` 对应)

- `ir/` —— 基础 IR、类型系统
- `tirx/` —— 新一代张量级 IR(见 `tirx-overview.md`)
- `s_tir/` —— 调度 IR(meta_schedule / dlight / tensor_intrin / backend)
- `relax/` —— Relax 新一代图级 IR,端到端流水线
- `topi/` —— 标准算子库
- `target/`、`backend/` —— 目标代码生成
- `runtime/` —— 推理运行时、RPC、设备抽象
- `arith/` —— 整数集与算术简化

近期活跃开发方向(从 git log):Relax IR、ONNX 前端、DLight 调度的修复与优化。

## 开发约定(来自 `AGENTS.md`)

- **构建**:优先复用已有 `build/`;新构建初始化子模块后用 CMake + Ninja。
- **不要 `pip install -e`**:必须用 `PYTHONPATH="$(pwd)/python:$(pwd)/.local/python"`,editable install 会让不同 worktree 串读代码。
- **测试**:先跑最小相关测试再扩大;`pre-commit` 在 PR 上只对改动文件运行。
- **提交信息 tag**:`[REFACTOR][IR]`、`[FIX][TIR]`、`[DOCS]` 等。
- **风格**:遵循周围代码风格,改动聚焦于任务本身,不做无关清理。
