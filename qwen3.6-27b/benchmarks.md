# Qwen3.6-27B Benchmarks in Single GB10 (Spark)

## Findings

- When using the default attention flashinfer for MTP, it throws `torch.AcceleratorError: CUDA error: an illegal memory access was encountered` randomly. Using `"attention_backend":"triton_attn"` in `speculative_config` solves the problem. However, it is slower than flashinfer but uses less VRAM.
- Changing `max_model_len` does NOT affect the benchmark results.
- Increasing `max_num_seqs` does NOT worsen the benchmark results, it sometimes improves the throughput.

## Benchmarks

```bash
nvidia/Qwen3.6-27B-NVFP4
--quantization modelopt \
--attention-backend flashinfer \
--speculative-config '{"method":"mtp","num_speculative_tokens":3}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 10070904965 # 1.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* unstable: speculative_attention_backend defaults to flashinfer, which errors randomly
* uses MarlinNvFp4LinearKernel for NVFP4 GEMM (W4A16)
* uses FlashInferFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 1060.2 │   30.2 │  1936.9 │    3 │
│     0 │    2 │ 1073.6 │   53.7 │  3749.4 │    3 │
│     0 │    5 │ 1140.7 │  113.0 │  8954.6 │    3 │
│     0 │   10 │ 1147.6 │  171.3 │ 17835.5 │    3 │
│  4096 │    1 │  481.2 │   30.7 │  4257.0 │    3 │
│  4096 │    2 │  490.0 │   57.8 │  8304.0 │    3 │
│  4096 │    5 │  512.0 │  109.3 │ 19969.8 │    3 │
│  4096 │   10 │  442.6 │   65.1 │ 38756.4 │    3 │
│  8192 │    1 │  541.2 │   29.7 │  3786.2 │    3 │
│  8192 │    2 │  555.2 │   56.0 │  7320.9 │    3 │
│  8192 │    5 │  578.9 │  115.9 │ 17657.8 │    3 │
│  8192 │   10 │  204.8 │   24.4 │ 66819.1 │    3 │
│ 16384 │    1 │  488.4 │   31.2 │  4194.6 │    3 │
│ 16384 │    2 │  497.2 │   51.3 │  8177.8 │    3 │
│ 16384 │    5 │  513.4 │  104.4 │ 19908.6 │    3 │

vllm version 0.26.1rc1.dev298+g1ea84d74b.d20260803
```

```bash
nvidia/Qwen3.6-27B-NVFP4
--quantization modelopt \
--speculative-config '{"method":"mtp","num_speculative_tokens":3}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 10070904965 # 1.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* probably unstable, but no crashes happened yet: speculative_attention_backend defaults to flashinfer, which can error randomly
* uses MarlinNvFp4LinearKernel for NVFP4 GEMM (W4A16)
* uses FlashInferFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃  ttfr ms ┃ runs ┃
│     0 │    1 │ 1040.8 │   28.1 │   1972.7 │    3 │
│     0 │    2 │ 1035.8 │   47.7 │   3876.5 │    3 │
│     0 │    5 │ 1126.7 │  107.1 │   9065.7 │    3 │
│     0 │   10 │ 1135.1 │  162.6 │  18033.2 │    3 │
│  4096 │    1 │  474.7 │   27.0 │   4316.1 │    3 │
│  4096 │    2 │  487.1 │   51.2 │   8349.9 │    3 │
│  4096 │    5 │  497.8 │  106.2 │  20539.8 │    3 │
│  4096 │   10 │  493.6 │  172.3 │  41468.5 │    3 │
│  8192 │    1 │  538.1 │   28.4 │   3807.3 │    3 │
│  8192 │    2 │  554.1 │   50.2 │   7331.8 │    3 │
│  8192 │    5 │  568.2 │  102.6 │  17987.8 │    3 │
│  8192 │   10 │  568.6 │  163.7 │  35994.8 │    3 │
│ 16384 │    1 │  481.0 │   28.9 │   4260.2 │    3 │
│ 16384 │    2 │  495.9 │   47.7 │   8196.2 │    3 │
│ 16384 │    5 │  510.2 │   98.4 │  20031.1 │    3 │
│ 16384 │   10 │  506.4 │  154.5 │  40407.1 │    3 │
│ 32768 │    1 │  391.9 │   25.3 │   5229.9 │    3 │
│ 32768 │    2 │  402.7 │   48.3 │  10106.1 │    3 │
│ 32768 │    5 │  411.3 │   95.8 │  24840.2 │    3 │
│ 32768 │   10 │   75.2 │    6.3 │ 172238.6 │    3 │
│ 65535 │    1 │  279.6 │   28.1 │   7327.9 │    3 │
│ 65535 │    2 │  284.9 │   46.5 │  14301.5 │    3 │
│ 65535 │    5 │  292.3 │   77.0 │  34961.8 │    3 │

vllm version 0.26.1rc1.dev298+g1ea84d74b.d20260803
```

```bash
nvidia/Qwen3.6-27B-NVFP4
--quantization modelopt \
--speculative-config '{"method":"mtp","num_speculative_tokens":3}' \

container: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 10070904965 # 1.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* probably unstable, but no crashes happened yet

// PENDING with nightly container

vllm version 0.26.1rc1.dev298+g1ea84d74b.d20260803
```

```bash
nvidia/Qwen3.6-27B-NVFP4
--quantization modelopt \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn"}' \

container: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 10070904965 # 1.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* stable

// PENDING with nightly container

vllm version 0.26.1rc1.dev298+g1ea84d74b.d20260803
```

```bash
nvidia/Qwen3.6-27B-NVFP4
--quantization modelopt \
--attention-backend flashinfer \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn"}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 10070904965 # 1.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* stable, but slower than the default flashinfer backend
* uses MarlinNvFp4LinearKernel for NVFP4 GEMM (W4A16)
* uses FlashInferFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 1080.1 │   27.2 │  1901.3 │    3 │
│     0 │    2 │ 1065.0 │   46.5 │  3818.3 │    3 │
│     0 │    5 │ 1127.9 │   85.7 │  8792.0 │    3 │
│     0 │   10 │ 1140.0 │  136.1 │ 17747.4 │    3 │
│  4096 │    1 │  479.8 │   28.4 │  4271.4 │    3 │
│  4096 │    2 │  490.9 │   45.6 │  7992.1 │    3 │
│  4096 │    5 │  503.7 │   94.7 │ 20143.5 │    3 │
│  4096 │   10 │  453.9 │   64.6 │ 37157.7 │    3 │
│  8192 │    1 │  532.7 │   28.1 │  3846.4 │    3 │
│  8192 │    2 │  552.0 │   46.5 │  7133.2 │    3 │
│  8192 │    5 │  562.1 │   82.3 │ 17447.3 │    3 │
│  8192 │   10 │  203.0 │   25.5 │ 70055.7 │    3 │
│ 16384 │    1 │  483.3 │   26.8 │  4239.0 │    3 │
│ 16384 │    2 │  491.5 │   43.8 │  7995.9 │    3 │
│ 16384 │    5 │  496.8 │   68.8 │ 20035.9 │    3 │


vllm version 0.26.1rc1.dev298+g1ea84d74b.d20260803
```

```bash
nvidia/Qwen3.6-27B-NVFP4
--quantization modelopt \
--attention-backend flashinfer \
--default-chat-template-kwargs '{"preserve_thinking":false}' \
--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":15}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  # kv_cache_memory: 10070904965 # error: 11.75 GiB KV cache is needed, which is larger than the available KV cache memory (9.38 GiB)
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* unknown stability
* uses more vram than mtp, requires disabled thinking
* uses MarlinNvFp4LinearKernel for NVFP4 GEMM (W4A16)
* uses FlashInferFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 1069.2 │   31.6 │  1920.1 │    3 │
│     0 │    2 │ 1081.2 │   50.8 │  3720.3 │    3 │
│     0 │    5 │ 1130.2 │   74.8 │  9015.0 │    3 │
│     0 │   10 │  739.8 │   49.2 │ 16279.9 │    3 │
│  4096 │    1 │  505.7 │   31.4 │  4053.4 │    3 │
│  4096 │    2 │  510.9 │   57.6 │  7947.6 │    3 │
│  4096 │    5 │  524.2 │   76.2 │ 19484.5 │    3 │
│  4096 │   10 │  353.2 │   27.8 │ 33403.6 │    3 │
│  8192 │    1 │  429.2 │   36.6 │  4775.0 │    3 │
│  8192 │    2 │  435.9 │   60.6 │  9326.4 │    3 │

vllm version 0.26.1rc1.dev298+g1ea84d74b.d20260803
```

```bash
sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP
--quantization modelopt \
--attention-backend flashinfer \
--speculative-config '{"method":"mtp","num_speculative_tokens":3}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 10070904965 # 1.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* unstable: speculative_attention_backend defaults to flashinfer, which errors randomly
* Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM (W4A4)

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 2273.8 │   23.6 │   904.4 │    3 │
│     0 │    2 │ 2258.3 │   44.3 │  1776.5 │    3 │
│     0 │    5 │ 2096.4 │   93.2 │  4860.0 │    3 │
│     0 │   10 │ 1572.6 │  154.3 │ 13007.6 │    3 │
│  4096 │    1 │  898.6 │   23.3 │  2281.0 │    3 │
│  4096 │    2 │  972.1 │   43.4 │  4168.6 │    3 │
│  4096 │    5 │  870.3 │   91.7 │ 11731.3 │    3 │
│  4096 │   10 │  697.2 │   86.6 │ 25293.4 │    3 │
│  8192 │    1 │  928.0 │   22.3 │  2209.0 │    3 │
│  8192 │    2 │ 1060.8 │   45.1 │  3792.9 │    3 │

vllm version 0.26.1rc1.dev298+g1ea84d74b.d20260803
```

```bash
unsloth/Qwen3.6-27B-NVFP4
--attention-backend flashinfer \
--speculative-config '{"method":"mtp","num_speculative_tokens":3}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 10070904965 # 1.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* unstable: speculative_attention_backend defaults to flashinfer, which errors randomly
* uses FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM (W4A4)
* uses CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 2065.9 │   25.3 │   995.6 │    3 │
│     0 │    2 │ 2033.3 │   44.7 │  1939.6 │    3 │
│     0 │    5 │ 1376.9 │  101.2 │  7410.3 │    3 │
│     0 │   10 │ 1266.7 │  145.8 │ 16166.2 │    3 │
│  4096 │    1 │  811.3 │   26.1 │  2525.6 │    3 │
│  4096 │    2 │  748.3 │   48.9 │  5410.4 │    3 │
│  4096 │    5 │  579.4 │  100.3 │ 17639.7 │    3 │
│  4096 │   10 │  513.9 │   69.2 │ 33952.7 │    3 │
│  8192 │    1 │  899.4 │   24.6 │  2280.0 │    3 │
│  8192 │    2 │  883.6 │   49.5 │  4571.2 │    3 │
│  8192 │    5 │  644.8 │  100.7 │ 15846.3 │    3 │

vllm version 0.26.1rc1.dev298+g1ea84d74b.d20260803
```
