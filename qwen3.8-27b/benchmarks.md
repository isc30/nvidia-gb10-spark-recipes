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
