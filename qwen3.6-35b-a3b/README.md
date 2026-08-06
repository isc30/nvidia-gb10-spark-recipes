# Qwen3.6-35B-A3B

- Recipe: [qwen3.6-35b-a3b.yaml](./recipe.yaml)
- [Benchmarks](./benchmarks.md)

## Model

- `nvidia/Qwen3.6-35B-A3B-NVFP4` provides the best accuracy and performance
- `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` same accuracy but much slower (the claims from unsloth do not apply to GB10 hardware).

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
