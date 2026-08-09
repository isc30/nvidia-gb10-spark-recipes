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

* UNSTABLE, cuda crashes, speculative_attention_backend defaults to flashinfer, which can error randomly
* uses MarlinNvFp4LinearKernel for NVFP4 GEMM (W4A16)
* uses FlashInferFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod

               total        used        free      shared  buff/cache   available
Mem:           121Gi        68Gi        30Gi       595Mi        23Gi        52Gi
Swap:           15Gi          0B        15Gi

┃  depth ┃ conc ┃ pp t/s ┃ tg t/s ┃  ttfr ms ┃ runs ┃
│      0 │    1 │ 1040.8 │   28.1 │   1972.7 │    3 │
│      0 │    2 │ 1035.8 │   47.7 │   3876.5 │    3 │
│      0 │    5 │ 1126.7 │  107.1 │   9065.7 │    3 │
│      0 │   10 │ 1135.1 │  162.6 │  18033.2 │    3 │
│   4096 │    1 │  474.7 │   27.0 │   4316.1 │    3 │
│   4096 │    2 │  487.1 │   51.2 │   8349.9 │    3 │
│   4096 │    5 │  497.8 │  106.2 │  20539.8 │    3 │
│   4096 │   10 │  493.6 │  172.3 │  41468.5 │    3 │
│   8192 │    1 │  538.1 │   28.4 │   3807.3 │    3 │
│   8192 │    2 │  554.1 │   50.2 │   7331.8 │    3 │
│   8192 │    5 │  568.2 │  102.6 │  17987.8 │    3 │
│   8192 │   10 │  568.6 │  163.7 │  35994.8 │    3 │
│  16384 │    1 │  481.0 │   28.9 │   4260.2 │    3 │
│  16384 │    2 │  495.9 │   47.7 │   8196.2 │    3 │
│  16384 │    5 │  510.2 │   98.4 │  20031.1 │    3 │
│  16384 │   10 │  506.4 │  154.5 │  40407.1 │    3 │
│  32768 │    1 │  391.9 │   25.3 │   5229.9 │    3 │
│  32768 │    2 │  402.7 │   48.3 │  10106.1 │    3 │
│  32768 │    5 │  411.3 │   95.8 │  24840.2 │    3 │
│  32768 │   10 │   75.2 │    6.3 │ 172238.6 │    3 │
│  65535 │    1 │  279.6 │   28.1 │   7327.9 │    3 │
│  65535 │    2 │  284.9 │   46.5 │  14301.5 │    3 │
│  65535 │    5 │  292.3 │   77.0 │  34961.8 │    3 │
│  65535 │   10 │   27.9 │    2.2 │ 460082.8 │    3 │
│ 100000 │    1 │  265.8 │   26.4 │   7706.7 │    3 │
│ 100000 │    2 │  275.3 │   41.6 │  14793.7 │    3 │
│ 100000 │    5 │   30.6 │    2.7 │ 201088.6 │    3 │

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
bottlecapai/ThinkingCap-Qwen3.6-27B-NVFP4
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn"}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 20042098000 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24

* stable

┃  depth ┃ conc ┃ pp t/s ┃ tg t/s ┃  ttfr ms ┃ runs ┃
│      0 │    1 │  893.9 │   22.3 │   2297.1 │    3 │
│      0 │    2 │  894.1 │   41.6 │   4404.2 │    3 │
│      0 │    5 │  943.2 │   79.2 │  10591.0 │    3 │
│      0 │   10 │  947.3 │  134.7 │  21435.3 │    3 │
│   4096 │    1 │  396.6 │   22.7 │   5166.0 │    3 │
│   4096 │    2 │  412.1 │   45.3 │   9752.1 │    3 │
│   4096 │    5 │  417.2 │   75.6 │  24218.0 │    3 │
│   4096 │   10 │  414.9 │  165.8 │  49360.0 │    3 │
│   8192 │    1 │  450.7 │   23.4 │   4546.0 │    3 │
│   8192 │    2 │  466.8 │   42.1 │   8621.9 │    3 │
│   8192 │    5 │  474.6 │   90.8 │  21420.0 │    3 │
│   8192 │   10 │  471.8 │  125.3 │  42007.8 │    3 │
│  16384 │    1 │  405.5 │   20.8 │   5052.3 │    3 │
│  16384 │    2 │  421.5 │   46.2 │   9716.1 │    3 │
│  16384 │    5 │  423.3 │   74.0 │  23087.1 │    3 │
│  16384 │   10 │  418.9 │  100.3 │  47728.2 │    3 │
│  32768 │    1 │  338.0 │   23.6 │   6062.2 │    3 │
│  32768 │    2 │  340.7 │   35.7 │  11336.8 │    3 │
│  32768 │    5 │  347.7 │   56.2 │  28768.0 │    3 │
│  32768 │   10 │   48.9 │    4.8 │ 219069.0 │    3 │
│  65535 │    1 │  241.0 │   21.5 │   8500.7 │    3 │
│  65535 │    2 │  241.7 │   34.4 │  16271.4 │    3 │
│  65535 │    5 │  249.4 │   53.5 │  40320.4 │    3 │
│  65535 │   10 │   25.6 │    2.0 │ 490718.3 │    3 │
│ 100000 │    1 │  233.4 │   21.0 │   8778.2 │    3 │
│ 100000 │    2 │  237.3 │   20.5 │  16246.4 │    3 │
│ 100000 │    5 │   19.4 │    1.6 │ 318939.1 │    3 │
│ 100000 │   10 │   15.0 │    1.1 │ 770912.8 │    3 │

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
sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP
--quantization modelopt \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn"}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 20042098000 # 1.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24

* stable
* Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM (W4A4)

┃  depth ┃ conc ┃ pp t/s ┃ tg t/s ┃  ttfr ms ┃ runs ┃
│      0 │    1 │ 2285.2 │   24.6 │    901.3 │    3 │
│      0 │    2 │ 2297.2 │   44.9 │   1755.4 │    3 │
│      0 │    5 │ 1977.4 │   95.0 │   5089.4 │    3 │
│      0 │   10 │ 1615.4 │  109.6 │  12365.8 │    3 │
│   4096 │    1 │  995.2 │   24.2 │   2059.8 │    3 │
│   4096 │    2 │ 1021.4 │   44.7 │   4009.4 │    3 │
│   4096 │    5 │  884.6 │  104.8 │  11573.9 │    3 │
│   4096 │   10 │  713.3 │   98.4 │  28249.9 │    3 │
│   8192 │    1 │  938.4 │   23.9 │   2184.9 │    3 │
│   8192 │    2 │ 1157.4 │   43.6 │   3538.2 │    3 │
│   8192 │    5 │  973.8 │   83.9 │  10241.1 │    3 │
│   8192 │   10 │  843.1 │  105.7 │  23432.5 │    3 │
│  16384 │    1 │  883.8 │   22.5 │   2322.8 │    3 │
│  16384 │    2 │  956.9 │   43.0 │   4097.7 │    3 │
│  16384 │    5 │  831.2 │   75.8 │  11832.9 │    3 │
│  16384 │   10 │  740.1 │   91.9 │  26070.5 │    3 │
│  32768 │    1 │  653.8 │   21.3 │   3135.3 │    3 │
│  32768 │    2 │  677.5 │   43.0 │   5905.9 │    3 │
│  32768 │    5 │  617.6 │   59.2 │  15828.8 │    3 │
│  32768 │   10 │  105.9 │    8.8 │ 115348.1 │    3 │
│  65535 │    1 │  419.1 │   22.2 │   4889.1 │    3 │
│  65535 │    2 │  413.3 │   35.5 │   9460.6 │    3 │
│  65535 │    5 │  388.6 │   54.0 │  25610.4 │    3 │
│  65535 │   10 │   40.9 │    3.0 │ 282860.1 │    3 │
│ 100000 │    1 │  353.0 │   20.0 │   5804.4 │    3 │

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

```bash
unsloth/Qwen3.6-27B-NVFP4
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn"}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 20042098000 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24

* stable
* uses FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM (W4A4)
* uses CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 2076.0 │   26.3 │   992.0 │    3 │
│     0 │    2 │ 1772.7 │   42.9 │  2275.6 │    3 │
│     0 │    5 │ 1467.9 │   84.4 │  6631.4 │    3 │
│     0 │   10 │ 1315.9 │  105.5 │ 15148.7 │    3 │
│  4096 │    1 │  845.2 │   25.5 │  2425.9 │    3 │
│  4096 │    2 │  727.5 │   47.5 │  5532.5 │    3 │
│  4096 │    5 │  606.8 │   90.6 │ 16602.4 │    3 │
│  4096 │   10 │  560.1 │  101.6 │ 35918.4 │    3 │
│  8192 │    1 │  917.5 │   26.2 │  2236.2 │    3 │
│  8192 │    2 │  886.2 │   48.2 │  4537.4 │    3 │
│  8192 │    5 │  675.4 │  101.6 │ 15159.9 │    3 │
│  8192 │   10 │  653.0 │  105.9 │ 29948.6 │    3 │
│ 16384 │    1 │  783.2 │   25.4 │  2617.3 │    3 │
│ 16384 │    2 │  738.5 │   48.9 │  5445.8 │    3 │
│ 16384 │    5 │  618.5 │   72.6 │ 15572.3 │    3 │
│ 16384 │   10 │  573.0 │   74.4 │ 34181.1 │    3 │
│ 32768 │    1 │  584.2 │   24.1 │  3507.6 │    3 │
│ 32768 │    2 │  559.4 │   37.9 │  6830.1 │    3 │

vllm version 0.26.1rc1.dev298+g1ea84d74b.d20260803
```
