# Qwen3.8-27B Benchmarks in Single GB10 (Spark)

## Benchmarks

```
recipe_version: '2'
max_nodes: 1
name: Qwen3.8-27B-NVFP4
model: RadixArk/Qwen3.8-27B-NVFP4
runtime: sglang
container: lmsysorg/sglang:dev-cu13
command: |
  python3 -m sglang.launch_server \
    --trust-remote-code --model-path {model} --tp-size {tensor_parallel} \
    --served-model-name {served_model_name} \
    --mem-fraction-static {mem_fraction_static} \
    --max-total-tokens {max_total_tokens} \
    --max-mamba-cache-size {max_mamba_cache_size} \
    --max-running-requests {max_running_requests} \
    --attention-backend {attention_backend} \
    --chunked-prefill-size {chunked_prefill_size} \
    --disable-prefill-cuda-graph \
    --cuda-graph-bs 1 2 3 4 5 8 10 12 \
    --speculative-algorithm {speculative_algorithm} \
    --speculative-draft-model-path {speculative_draft_model_path} \
    --speculative-num-draft-tokens {speculative_num_draft_tokens} \
    --mamba-radix-cache-strategy extra_buffer \
    --enable-torch-compile --torch-compile-max-bs {torch_compile_max_bs} \
    --num-continuous-decode-steps {num_continuous_decode_steps} \
    --reasoning-parser {reasoning_parser} --tool-call-parser {tool_call_parser} \
    --watchdog-timeout {watchdog_timeout} \
    --host {host} --port {port}
defaults:
  port: 8001
  host: 0.0.0.0
  tensor_parallel: 1
  served_model_name: qwen3-27b
  attention_backend: flashinfer
  tool_call_parser: qwen3_coder
  mem_fraction_static: 0.45
  max_total_tokens: 152909 # 262144
  max_mamba_cache_size: 64
  max_running_requests: 12
  chunked_prefill_size: 8192
  speculative_algorithm: DFLASH
  speculative_draft_model_path: incoai/Qwen3.8-27B-DFlash2
  speculative_num_draft_tokens: 8
  num_continuous_decode_steps: 2
  torch_compile_max_bs: 10
  reasoning_parser: qwen3
  watchdog_timeout: 600

┃  depth ┃ conc ┃ pp t/s ┃ tg t/s ┃  ttfr ms ┃ runs ┃
│      0 │    1 │ 2332.0 │   26.5 │    882.8 │    3 │
│      0 │    2 │ 2163.3 │   49.5 │   1895.3 │    3 │
│      0 │    5 │ 2066.3 │   90.4 │   4916.4 │    3 │
│      0 │   10 │ 2027.4 │  120.9 │   9505.6 │    3 │
│   4096 │    1 │  693.6 │   31.5 │   2955.8 │    3 │
│   4096 │    2 │  856.8 │   52.0 │   4778.1 │    3 │
│   4096 │    5 │  883.1 │   56.8 │  10992.0 │    3 │
│   4096 │   10 │  613.3 │   42.1 │  19513.6 │    3 │
│   8192 │    1 │  364.9 │   30.1 │   5616.4 │    3 │
│   8192 │    2 │  607.1 │   46.3 │   6737.6 │    3 │
│   8192 │    5 │  914.7 │   57.2 │   9809.1 │    3 │
│   8192 │   10 │  330.5 │   21.4 │  36307.5 │    3 │
│  16384 │    1 │  320.7 │   29.6 │   6388.6 │    3 │
│  16384 │    2 │  317.9 │   49.5 │  12873.7 │    3 │
│  16384 │    5 │  174.4 │   13.2 │  34457.5 │    3 │
│  16384 │   10 │  159.4 │   10.9 │  69330.0 │    3 │
│  32768 │    1 │  266.4 │   25.3 │   7691.7 │    3 │
│  32768 │    2 │  132.2 │   10.0 │  20310.9 │    3 │
│  32768 │    5 │   84.7 │    5.7 │  65879.8 │    3 │
│  32768 │   10 │   74.8 │    4.8 │ 148175.3 │    3 │
```

```
recipe_version: '1'
model: RadixArk/Qwen3.8-27B-NVFP4
builder: isc30
container: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
build_args:
  - '--apply-vllm-pr'
  - '52816'
mods:
  - mods/dflash2-nvfp4-lmhead
defaults:
  served_model_name: qwen3-27b
  port: 8001
  host: 0.0.0.0
  tensor_parallel: 1
  gpu_memory_utilization: 0.3 # KEEP LINE ONLY IF SINGLE-MODEL
  kv_cache_memory: 20042098000 # KEEP LINE ONLY IF MULTI-MODEL # 1.0x=9971193034
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24
command: |
  vllm serve RadixArk/Qwen3.8-27B-NVFP4 \
    --host {host} \
    --port {port} \
    --served-model-name {served_model_name} \
    --tensor-parallel-size {tensor_parallel} \
    --optimization-level 3 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --load-format safetensors \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --kv-cache-memory {kv_cache_memory} \
    --max-model-len {max_model_len} \
    --max-num-batched-tokens {max_num_batched_tokens} \
    --max-num-seqs {max_num_seqs} \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --skip-mm-profiling \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --default-chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
    --generation-config auto \
    --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":8}' \
env:
  # defaults for gb10
  CUTE_DSL_ARCH: sm_121a
  TORCH_CUDA_ARCH_LIST: 12.1a
  ENABLE_NVFP4_SM100: '0'
  PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'
  # vllm
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE: '600'
  CUDA_MODULE_LOADING: LAZY
  FLASHINFER_DISABLE_VERSION_CHECK: '1'
  # cache
  VLLM_CACHE_ROOT: /cache/huggingface/vllm-cache
  TRITON_CACHE_DIR: /cache/huggingface/triton-cache
  TORCHINDUCTOR_CACHE_DIR: /cache/huggingface/torchinductor-cache
  TORCHINDUCTOR_FX_GRAPH_CACHE: '1'
  # build
  MAX_JOBS: '4'
  NVCC_THREADS: '2'
  FLASHINFER_NVCC_THREADS: '2'


```
