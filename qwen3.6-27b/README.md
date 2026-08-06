# Qwen3.6-27B

- Recipe: [qwen3.6-27b.yaml](./recipe.yaml)
- [Benchmarks](./benchmarks.md)

## Model

- `nvidia/Qwen3.6-27B-NVFP4` provides the best accuracy and performance (1060-1080 t/s prefill at depth 0)
- `sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP` is faster (2273 t/s prefill) but may have different behavior
- `unsloth/Qwen3.6-27B-NVFP4` is comparable to nvidia's version with similar performance (2065 t/s prefill)

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
