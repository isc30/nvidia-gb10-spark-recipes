# Qwen3.6-27B

- Recipe: [qwen3.6-27b.yaml](./recipe.yaml)
- [Benchmarks](./benchmarks.md)

## Model

- `nvidia/Qwen3.6-27B-NVFP4` provides the best accuracy and performance (1060-1080 t/s prefill at depth 0)
- `sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP` has less latency but the token generation is slower. Its also abliterated so the accuracy is impacted too.
- `unsloth/Qwen3.6-27B-NVFP4` is comparable to nvidia's version with similar performance but better latency.

## Performance Comparison

| scenario | metric | nvidia | huihui | unsloth |
|----------|--------|--------|--------|---------|
| depth=0, c=1 | prefill | 🔴 1040.8 | 🟢 2273.8 | 🟡 2065.9 |
| | tg | 🟢 28.1 | 🔴 23.6 | 🟡 25.3 |
| | ttfr | 🟡 1972.7 | 🟢 904.4 | 🔴 995.6 |
| depth=0, c=5 | prefill | 🔴 1126.7 | 🟢 2096.4 | 🟡 1376.9 |
| | tg | 🟢 107.1 | 🔴 93.2 | 🟡 101.2 |
| | ttfr | 🔴 9065.7 | 🟢 4860.0 | 🟡 7410.3 |
| depth=0, c=10 | prefill | 🔴 1135.1 | 🟢 1572.6 | 🟡 1266.7 |
| | tg | 🟢 162.6 | 🟡 154.3 | 🔴 145.8 |
| | ttfr | 🔴 18033.2 | 🟢 13007.6 | 🟡 16166.2 |
| depth=4096, c=1 | prefill | 🔴 474.7 | 🟢 898.6 | 🟡 811.3 |
| | tg | 🟢 27.0 | 🔴 23.3 | 🟡 26.1 |
| | ttfr | 🔴 4316.1 | 🟢 2281.0 | 🟡 2525.6 |
| depth=4096, c=5 | prefill | 🟡 497.8 | 🟢 870.3 | 🔴 579.4 |
| | tg | 🟢 106.2 | 🟡 91.7 | 🔴 100.3 |
| | ttfr | 🔴 20539.8 | 🟢 11731.3 | 🟡 17639.7 |
| depth=4096, c=10 | prefill | 🟡 493.6 | 🟢 697.2 | 🔴 513.9 |
| | tg | 🟢 172.3 | 🔴 86.6 | 🟡 69.2 |
| | ttfr | 🔴 41468.5 | 🟢 25293.4 | 🟡 33952.7 |
| depth=8192, c=1 | prefill | 🔴 538.1 | 🟢 928.0 | 🟡 899.4 |
| | tg | 🟢 28.4 | 🔴 22.3 | 🟡 24.6 |
| | ttfr | 🔴 3807.3 | 🟢 2209.0 | 🟡 2280.0 |
| depth=8192, c=5 | prefill | 🔴 568.2 | 🟢 870.3 | 🟡 644.8 |
| | tg | 🟢 102.6 | 🟡 91.7 | 🔴 100.7 |
| | ttfr | 🔴 17987.8 | 🟢 11731.3 | 🟡 15846.3 |

## Model Parameters

### Main model Attention backends
- `flashinfer` the fastest and stable.
- `triton_attn` slower than `flashinfer` and also stable.

Recommendation: `--attention-backend flashinfer`

### MTP model Attention backends
- `triton_attn` is the 2nd fastest but stable.
- `flashinfer` the fastest but **NOT stable**.

Recommendation: `--speculative-config: {"attention_backend":"triton_attn"}`

### Speculative config
- `mtp` with `num_speculative_tokens=3` is the fastest and stable.
- `dflash` (`z-lab/Qwen3.6-27B-DFlash`, 15 speculative tokens)  is slower than MTP in all tests.

Recommendation: `--speculative-config: {"method":"mtp","num_speculative_tokens":3}`

## Stability Tweaks

- `mods/fix-qwen3.6-chat-template` + `--chat-template fixed_chat_template.jinja` + `--tool-call-parser qwen3_xml` are required to generate the proper tool calls and get the model running as expected without failing to run tools or cutting responses abruptly.
- `--override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty":0.0, "repetition_penalty":1.1}'` is required to prevent looping and matches the official qwen recommendations.
