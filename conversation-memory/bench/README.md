# DSA 基线 bench 脚本(P0.4)

薄封装,黑盒调用各基线库的公开 API(参考纪律:不读底层 kernel 实现)。数字结果见同目录 `*_baseline.json`,解读见上一级 [[baselines.md]]。

## 运行(环境变量)H200

```bash
export PYTHONPATH=/root/code/tvm/.local/python:/root/code/tvm/python
export TVM_LIBRARY_PATH=/root/code/tvm/build/lib
python3 bench_indexer.py        # deep_gemm fp8 paged MQA logits(lightning indexer)
python3 bench_mla_decode.py     # sgl_kernel FlashMLA dense MLA decode
```

口径:CUDA Event,warmup 3–5 / iter 15–20,取中位数 ms。

## 文件

| 文件 | 组件 | 依赖 |
| --- | --- | --- |
| `bench_indexer.py` | A:lightning indexer | `deep_gemm`(预装;旧 API) |
| `bench_mla_decode.py` | B:dense MLA decode | `sgl_kernel.flash_mla`(sglang 预编译) |

组件 C(sparse MLA)基线探针见 [[baselines.md]] 记录(`flash_mla_sparse_fwd`,SM90 可用),脚本待 P2 补齐。
