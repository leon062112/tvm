# H200 开发环境备忘(构建 / 运行 / 网络 / 基线栈)

- **主题**:在这台 H200 机器上构建、运行 tirx 与各 DSA 基线所需的全部环境事实与坑
- **时间**:2026-09-09(P0 期间沉淀)
- **来源**:本机实操;相关的计划/基线见 [[dsa-reproduction-plan.md]]、[[baselines.md]]

## 硬件识别(重要)

- `nvidia-smi` 报 **"NVIDIA L20X"、compute_cap 8.9** ——**都是掩码假象**。
- 真实:**sm_90a Hopper / H200 级**,证据三合一:torch `get_device_capability()`=(9,0)、132 SM、139.8 GiB 可用、PCIe、driver 570.148.08、CUDA 12.8。
- **一律用 `torch.cuda.get_device_capability()` / pynvml 判 compute capability,不要信 `nvidia-smi` 的 compute_cap 列。**

## 构建 TVM + tirx

```bash
git submodule update --init --recursive
mkdir -p build && cp cmake/config.cmake build/config.cmake
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=ON
cmake --build build --parallel        # 192 核约 10 分钟
```

- **tvm_ffi 必须用 submodule 版**:site-packages 里的旧版 import 会报 `cannot import name 'load_lib_ctypes'`。装法(非 editable,符合 AGENTS.md):
  `pip install --target=$(pwd)/.local/python $(pwd)/3rdparty/tvm-ffi`
- 冒烟 + 运行环境变量:
  ```bash
  export PYTHONPATH=/root/code/tvm/.local/python:/root/code/tvm/python:/root/code/tirx-kernels
  export TVM_LIBRARY_PATH=/root/code/tvm/build/lib
  python3 -c "import tvm, tvm.tirx; print(tvm.__version__)"
  ```
- 额外依赖:`pip install ml_dtypes`(不装会导致大量测试收集时报 ModuleNotFoundError)。

## 测试面(H200 上有结构性缺口)

- `tests/python/tirx/operator/tile_primitive/cuda/conftest.py` 用 `env.has_cuda_compute(10)` 把整个 CUDA tile-primitive 套件 **skip 到 cc≥10** → H200(cc9.0)上这些 GPU 测试**全部 skip**。
- 实测可跑:编译器侧(parser/printer/layout/exec_scope/verifier/transform)全过;`operator/` 143 passed / 1564 skipped;`codegen/` 154 passed / 320 skipped / **9 failed(PTX-dialect/tensor-map 侧,预存在,源码零改动)**。
- 结论:**GPU kernel 正确性语料要靠自己积累**(从 P1 第一个 WGMMA 开始);这就是 [[tirx-harness-gap.md]] 卡点②的实证。

## 网络(代理环境)

- 存在 `AISTUDIO_PROXY_ADDR` 代理。
- 可达:**PyPI**、**github.com / api.github.com**、pypi JSON。
- 不可达:`raw.githubusercontent.com`、`conda.anaconda.org`(conda 报 URL 被剥冒号成 `https//`,**conda 基本不可用,换 gcc 走不通**)、`edge.urm.nvidia.com`(flashinfer 拉 cubin 失败)。

## 基线栈现状(DSA 用)

| 库 | 状态 | 备注 |
| --- | --- | --- |
| **deep_gemm** | ✅ 预装,sm90 可用 | 旧 API:位置参数、`fp8_paged_mqa_logits(q, kv, w, ctx, bt, meta, maxctx, clean)`、metadata 3 参数无 indices、`next_n` 仅 1/2 |
| **sgl_kernel.flash_mla** | ✅ **预编译 FlashMLA,sm90 可用,本项目的 dense+sparse MLA 基线** | `flash_mla_with_kvcache`(dense;sparse 需 indices)、`flash_mla_sparse_fwd`(sparse,SM90 通,topk 须 128 倍数)、`get_mla_metadata` |
| vllm `_flashmla_C.so` | ⚠️ | 有,但 numpy 2.2.6 二进制不兼容,import 报 dtype size changed |
| flashinfer 0.4.1 | ❌ MLA | MLA decode 构建仅 SM80;trtllm-gen 需拉 NVIDIA cubin(网络不通) |
| sgl_kernel `cutlass_mla_decode` | ❌ | **SM100-only**(`only supported on cc 10.0`) |
| FlashMLA 源码编译 | ❌ | sm100 需 NVCC≥12.9;gcc 10.2.1 编 CUTLASS 头 **cc1plus 段错误**(编译器 bug,非 OOM)→ 改用 sgl_kernel 预编译版 |

## 工具链

- python 3.10.13(/opt/conda)、gcc 10.2.1(**过旧,编 CUTLASS-heavy 代码会段错误**;系统无 gcc-toolset,yum/conda 都装不上)、cmake 3.26.4、ninja 1.11.1、CUDA 12.8、无 ccache。
- 预装:torch 2.8.0+cu128、triton 3.4.0、flashinfer 0.4.1、sglang 0.5.4、vllm 0.11.0、sgl-kernel 0.3.16、deep_gemm、cutlass-dsl 4.2.1。

## 工作区

- TVM 仓库:`/root/code/tvm`(fork 自 code.alipay.com/denghaodong.dhd/tvm)
- tirx-kernels:`/root/code/tirx-kernels`(github.com/mlc-ai/tirx-kernels)
- 基线 bench 脚本:[[bench/]] 与 `/root/code/dsa-bench/`

[[dsa-reproduction-plan.md]] [[baselines.md]] [[shapes.md]]
