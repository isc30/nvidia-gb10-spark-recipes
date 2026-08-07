# Qwen3.6-35B-A3B

- Recipe: [qwen3.6-35b-a3b.yaml](./recipe.yaml)
- [Benchmarks](./benchmarks.md)

## Model

- `nvidia/Qwen3.6-35B-A3B-NVFP4` provides the best accuracy and performance
- `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` same accuracy but with comparable performance.
- `THe-Plague/Qwen3.6-35B-A3B-abliterated-NVFP4-MTP` (huihui) is slightly faster but abliterated so accuracy is impacted.

## Performance Comparison

| scenario | metric | nvidia | huihui | unsloth |
|----------|--------|--------|--------|---------|
| depth=0, c=1 | prefill | 🟢 5714.8 | 🔴 5378.1 | 🟡 5487.0 |
| | tg | 🟢 101.3 | 🔴 77.3 | 🟡 94.8 |
| | ttfr | 🟢 362.2 | 🟡 374.9 | 🔴 384.2 |
| depth=0, c=5 | prefill | 🟡 6475.9 | 🟢 6553.9 | 🔴 6298.7 |
| | tg | 🔴 185.7 | 🟡 195.9 | 🟢 221.4 |
| | ttfr | 🟢 1467.5 | 🟡 1495.9 | 🔴 1613.9 |
| depth=0, c=10 | prefill | 🟢 6883.3 | 🟡 6759.9 | 🔴 6008.6 |
| | tg | 🔴 280.1 | 🟡 278.4 | 🟢 341.8 |
| | ttfr | 🟢 2865.1 | 🟡 2867.2 | 🔴 3401.9 |
| depth=4096, c=1 | prefill | 🟢 1999.3 | 🟡 1948.9 | 🔴 1984.5 |
| | tg | 🟢 97.3 | 🔴 63.0 | 🟡 92.3 |
| | ttfr | 🟢 1026.1 | 🟡 1034.4 | 🔴 1053.7 |
| depth=4096, c=5 | prefill | 🟢 2247.6 | 🟡 2141.2 | 🔴 1958.2 |
| | tg | 🔴 165.3 | 🟡 166.6 | 🟢 220.2 |
| | ttfr | 🟢 4337.8 | 🟡 4506.2 | 🔴 5214.6 |
| depth=4096, c=10 | prefill | 🟢 2269.0 | 🟡 2239.5 | 🔴 2002.1 |
| | tg | 🔴 208.3 | 🟡 209.6 | 🟢 317.6 |
| | ttfr | 🟡 8609.4 | 🟢 8539.0 | 🔴 10131.8 |
| depth=8192, c=1 | prefill | 🟡 1873.8 | 🔴 1825.4 | 🟡 1873.8 |
| | tg | 🟢 98.0 | 🔴 68.0 | 🟡 87.5 |
| | ttfr | 🟡 1094.0 | 🔴 1125.9 | 🟡 1094.0 |
| depth=8192, c=5 | prefill | 🟢 2099.0 | 🟡 2080.2 | 🔴 1864.2 |
| | tg | 🔴 167.6 | 🟡 144.6 | 🟢 216.6 |
| | ttfr | 🟢 4644.9 | 🟡 4698.3 | 🔴 5476.9 |
| depth=16384, c=1 | prefill | 🟡 1697.9 | 🔴 1651.9 | 🟢 1699.3 |
| | tg | 🟢 97.4 | 🔴 66.1 | 🟡 83.7 |
| | ttfr | 🟡 1208.0 | 🔴 1241.3 | 🟢 1207.2 |
| depth=16384, c=5 | prefill | 🟢 1926.0 | 🟡 1857.8 | 🔴 1727.6 |
| | tg | 🟢 234.2 | 🔴 133.6 | 🟡 208.4 |
| | ttfr | 🟡 5296.1 | 🟢 5259.4 | 🔴 5906.0 |
| depth=16384, c=10 | prefill | 🟢 1952.4 | 🟡 1900.3 | 🔴 1763.2 |
| | tg | 🔴 205.1 | 🟡 162.4 | 🟢 243.4 |
| | ttfr | 🟡 10183.7 | 🟢 10069.3 | 🔴 11431.6 |
| depth=32768, c=1 | prefill | 🟡 1488.9 | 🔴 1339.1 | 🟢 1498.2 |
| | tg | 🟢 98.2 | 🔴 54.9 | 🟡 91.9 |
| | ttfr | 🟡 1377.1 | 🔴 1532.7 | 🟢 1368.6 |

## Model Parameters

### Main model Attention backends
- `flashinfer` the fastest and stable.
- `triton_attn` slower than `flashinfer` and also stable.

Recommendation: `--attention-backend flashinfer`

### MTP model Attention backends
- `triton_attn` is the 2nd fastest but stable.
- `flashinfer` the fastest but **NOT stable**.

Recommendation: `--speculative-config: {"attention_backend":"triton_attn"}`

### Main model MoE backends
- `marlin`+ `VLLM_MARLIN_USE_ATOMIC_ADD` is 2nd fastest and uses very little VRAM. Use it when running multiple models in the same machine.
- `flashinfer_b12x` is the fastest (+10% marlin) but uses a lot of VRAM. Use it when just running this model on the machine.

Recommendation: either, depending on your VRAM needs. For example `--moe-backend marlin`

### MTP model MoE backends
- `triton` is the fastest and most stable.

Recommendation: `--speculative-config: {"moe_backend":"triton"}`

### Speculative config
- `mtp` with `num_speculative_tokens=3` is the fastest and stable.
- `dflash` (`z-lab/Qwen3.6-35B-A3B-DFlash`, 16 speculative tokens) is slower than MTP in all tests.

Recommendation: `--speculative-config: {"method":"mtp","num_speculative_tokens":3}`

## Stability Tweaks

- `mods/fix-qwen3.6-chat-template` + `--chat-template fixed_chat_template.jinja` + `--tool-call-parser qwen3_xml` are required to generate the proper tool calls and get the model running as expected without failing to run tools or cutting responses abruptly.
- `--override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty":0.0, "repetition_penalty":1.1}'` is required to prevent looping and matches the official qwen recommendations.
