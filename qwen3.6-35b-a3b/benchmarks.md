# Qwen3.6-35B-A3B Benchmarks in Single GB10 (Spark)

## Findings

- When using the default attention flashinfer for MTP, it throws `torch.AcceleratorError: CUDA error: an illegal memory access was encountered` randomly. Using `"attention_backend":"triton_attn"` in `speculative_config` solves the problem. However, it is slower than flashinfer but uses less VRAM.
- Changing `max_model_len` does NOT affect the benchmark results.
- Increasing `max_num_seqs` does NOT worsen the benchmark results, it sometimes improves the throughput.

## Benchmarks

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--quantization modelopt \
--attention-backend flashinfer \
--moe-backend marlin \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.3
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* best performance for the vram cost
* unstable (speculative_attention_backend defaults to flashinfer, which throws)
  > torch.AcceleratorError: CUDA error: an illegal memory access was encountered

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5371.1 │   97.0 │   383.8 │    3 │
│     0 │    2 │ 4770.8 │  139.6 │   855.5 │    3 │
│     0 │    5 │ 6477.9 │  231.6 │  1569.3 │    3 │
│     0 │   10 │ 6848.2 │  363.6 │  2983.8 │    3 │
│  4096 │    1 │ 1974.4 │   97.8 │  1038.5 │    3 │
│  4096 │    2 │ 2102.0 │  144.7 │  1931.0 │    3 │
│  4096 │    5 │ 2256.4 │  236.5 │  4524.1 │    3 │
│  4096 │   10 │ 2283.9 │  313.2 │  8870.2 │    3 │
│  8192 │    1 │ 1867.8 │  102.4 │  1097.8 │    3 │
│  8192 │    2 │ 1974.3 │  147.8 │  2056.0 │    3 │
│  8192 │    5 │ 2124.0 │  234.2 │  4805.3 │    3 │
│  8192 │   10 │ 2148.9 │  330.5 │  9468.7 │    3 │
│ 16384 │    1 │ 1696.2 │  101.1 │  1208.4 │    3 │
│ 16384 │    2 │ 1790.4 │  157.4 │  2267.4 │    3 │
│ 16384 │    5 │ 1931.6 │  221.7 │  5281.5 │    3 │
│ 16384 │   10 │ 1959.6 │  241.7 │ 10265.0 │    3 │
│ 32768 │    1 │ 1499.4 │   97.7 │  1368.1 │    3 │
│ 32768 │    2 │ 1573.4 │  147.0 │  2581.4 │    3 │

vllm version 0.23.1rc1.dev1302+ge765bbc97.d20260720
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--quantization modelopt \
--attention-backend flashinfer \
--moe-backend marlin \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  # gpu_memory_utilization: 0.3
  kv_cache_memory: 13438145445 # 7.22x
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* best performance for the vram cost
* unstable (speculative_attention_backend defaults to flashinfer, which throws)
  > torch.AcceleratorError: CUDA error: an illegal memory access was encountered

               total        used        free      shared  buff/cache   available
Mem:           121Gi        59Gi        38Gi       214Mi        25Gi        62Gi
Swap:           15Gi       567Mi        15Gi

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5417.5 │   91.7 │   380.4 │    3 │
│     0 │    2 │ 4980.2 │  141.9 │   814.6 │    3 │
│     0 │    5 │ 6492.0 │  237.0 │  1565.8 │    3 │
│     0 │   10 │ 6843.0 │  355.5 │  2986.5 │    3 │
│  4096 │    1 │ 1973.5 │  104.3 │  1039.4 │    3 │
│  4096 │    2 │ 2101.6 │  154.5 │  1931.2 │    3 │
│  4096 │    5 │ 2230.0 │  230.8 │  4578.8 │    3 │
│  4096 │   10 │ 2257.9 │  313.9 │  8975.5 │    3 │
│  8192 │    1 │ 1854.8 │  107.4 │  1106.1 │    3 │
│  8192 │    2 │ 1974.1 │  152.7 │  2056.6 │    3 │
│  8192 │    5 │ 2121.7 │  238.2 │  4809.6 │    3 │
│  8192 │   10 │ 2150.0 │  291.1 │  9390.1 │    3 │
│ 16384 │    1 │ 1697.9 │   97.4 │  1208.0 │    3 │
│ 16384 │    2 │ 1788.4 │  156.0 │  2271.4 │    3 │
│ 16384 │    5 │ 1926.0 │  234.2 │  5296.1 │    3 │
│ 16384 │   10 │ 1952.4 │  205.1 │ 10183.7 │    3 │
│ 32768 │    1 │ 1488.9 │   98.2 │  1377.1 │    3 │
│ 32768 │    2 │ 1584.4 │  144.1 │  2564.4 │    3 │

vllm version 0.23.1rc1.dev1302+ge765bbc97.d20260720
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--quantization modelopt \
--attention-backend flashinfer \
--moe-backend marlin \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn","moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.3
  kv_cache_memory: 13438145445 # 7.22x
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* the most stable, performance and vram-usage effective
* selected winner

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5714.8 │  101.3 │   362.2 │    3 │
│     0 │    2 │ 4911.5 │  141.1 │   781.2 │    3 │
│     0 │    5 │ 6475.9 │  185.7 │  1467.5 │    3 │
│     0 │   10 │ 6883.3 │  280.1 │  2865.1 │    3 │
│  4096 │    1 │ 1999.3 │   97.3 │  1026.1 │    3 │
│  4096 │    2 │ 2096.5 │  125.5 │  1790.6 │    3 │
│  4096 │    5 │ 2247.6 │  165.3 │  4337.8 │    3 │
│  4096 │   10 │ 2269.0 │  208.3 │  8609.4 │    3 │
│  8192 │    1 │ 1871.0 │   98.0 │  1095.7 │    3 │
│  8192 │    2 │ 1967.5 │  125.4 │  1909.1 │    3 │
│  8192 │    5 │ 2099.0 │  167.6 │  4644.9 │    3 │
│  8192 │   10 │ 2121.5 │  207.2 │  9258.5 │    3 │
│ 16384 │    1 │ 1670.9 │   96.8 │  1226.9 │    3 │
│ 16384 │    2 │ 1766.3 │  116.9 │  2129.9 │    3 │
│ 16384 │    5 │ 1893.1 │  154.0 │  5152.7 │    3 │
│ 16384 │   10 │ 1907.6 │  140.1 │  9568.7 │    3 │
│ 32768 │    1 │ 1482.0 │   85.8 │  1383.7 │    3 │
│ 32768 │    2 │ 1547.9 │  111.7 │  2444.3 │    3 │

vllm version 0.23.1rc1.dev1302+ge765bbc97.d20260720
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--quantization modelopt \
--attention-backend triton_attn \
--moe-backend marlin \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn","moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.3
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* best for little vram usage (triton < flashinfer < flashinfer_b12x)

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5620.0 │  104.4 │   368.4 │    3 │
│     0 │    2 │ 4623.9 │  138.8 │   947.4 │    3 │
│     0 │    5 │ 6428.0 │  193.5 │  1478.2 │    3 │
│     0 │   10 │ 6769.3 │  284.8 │  2913.3 │    3 │
│  4096 │    1 │ 1948.4 │   89.1 │  1053.3 │    3 │
│  4096 │    2 │ 2011.6 │  125.1 │  1864.4 │    3 │
│  4096 │    5 │ 2130.1 │  164.2 │  4574.2 │    3 │
│  4096 │   10 │ 2151.9 │  204.1 │  9075.7 │    3 │
│  8192 │    1 │ 1716.4 │   88.6 │  1194.8 │    3 │
│  8192 │    2 │ 1771.7 │  121.9 │  2119.0 │    3 │
│  8192 │    5 │ 1877.4 │  157.1 │  5192.5 │    3 │
│  8192 │   10 │ 1893.4 │  195.9 │ 10370.7 │    3 │
│ 16384 │    1 │ 1419.6 │   80.4 │  1443.7 │    3 │
│ 16384 │    2 │ 1470.5 │  105.3 │  2557.8 │    3 │
│ 16384 │    5 │ 1552.0 │  144.3 │  6290.0 │    3 │
│ 16384 │   10 │ 1563.3 │  139.2 │ 11816.3 │    3 │
│ 32768 │    1 │ 1124.8 │   65.8 │  1822.5 │    3 │
│ 32768 │    2 │ 1159.1 │   88.7 │  3268.6 │    3 │
│ 32768 │    5 │ 1220.2 │  120.4 │  8035.4 │    3 │

vllm version 0.23.1rc1.dev1302+ge765bbc97.d20260720
```

```bash
THe-Plague/Qwen3.6-35B-A3B-abliterated-NVFP4-MTP
--attention-backend FLASHINFER \
--moe-backend FLASHINFER_CUTLASS \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASHINFER","moe_backend":"FlashInfer CUTLASS"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 13438145445 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5547.8 │   74.7 │   371.1 │    3 │
│     0 │    2 │ 5070.8 │  117.6 │   785.8 │    3 │
│     0 │    5 │ 6889.0 │  206.6 │  1473.6 │    3 │
│     0 │   10 │ 7312.8 │  368.3 │  2791.9 │    3 │
│  4096 │    1 │ 2116.9 │   71.7 │   969.0 │    3 │
│  4096 │    2 │ 2263.9 │  115.4 │  1786.7 │    3 │
│  4096 │    5 │ 2409.1 │  185.7 │  4233.3 │    3 │
│  4096 │   10 │ 2464.2 │  280.9 │  8218.4 │    3 │
│  8192 │    1 │ 1969.3 │   66.3 │  1042.2 │    3 │
│  8192 │    2 │ 2110.7 │  113.4 │  1917.2 │    3 │
│  8192 │    5 │ 2244.2 │  185.4 │  4542.0 │    3 │
│  8192 │   10 │ 2297.1 │  277.8 │  8872.1 │    3 │
│ 16384 │    1 │ 1628.6 │   60.4 │  1261.1 │    3 │
│ 16384 │    2 │ 1881.1 │   92.1 │  2175.3 │    3 │
│ 16384 │    5 │ 1987.4 │  173.2 │  5115.2 │    3 │
│ 16384 │   10 │ 2142.5 │  279.9 │  9530.2 │    3 │
│ 32768 │    1 │ 1509.4 │   62.7 │  1357.9 │    3 │
│ 32768 │    2 │ 1642.6 │  112.9 │  2465.6 │    3 │
│ 32768 │    5 │ 1768.1 │  174.9 │  5757.0 │    3 │

vllm version 0.26.1rc1.dev371+g85ea44b46.d20260805
```

```bash
THe-Plague/Qwen3.6-35B-A3B-abliterated-NVFP4-MTP
--attention-backend FLASHINFER \
--moe-backend MARLIN \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn","moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 13438145445 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24

* same config as the "winner" nvidia one (most stable and performant)

┃  depth ┃ conc ┃ pp t/s ┃ tg t/s ┃  ttfr ms ┃ runs ┃
│      0 │    1 │ 5378.1 │   77.3 │    384.2 │    3 │
│      0 │    2 │ 5182.9 │  126.1 │    803.8 │    3 │
│      0 │    5 │ 6553.9 │  195.9 │   1495.9 │    3 │
│      0 │   10 │ 6759.9 │  278.4 │   2867.2 │    3 │
│   4096 │    1 │ 1948.9 │   63.0 │   1053.7 │    3 │
│   4096 │    2 │ 2002.7 │   92.8 │   1971.9 │    3 │
│   4096 │    5 │ 2141.2 │  166.6 │   4506.2 │    3 │
│   4096 │   10 │ 2239.5 │  209.6 │   8539.0 │    3 │
│   8192 │    1 │ 1825.4 │   68.0 │   1125.9 │    3 │
│   8192 │    2 │ 1922.5 │  103.1 │   1974.5 │    3 │
│   8192 │    5 │ 2080.2 │  144.6 │   4698.3 │    3 │
│   8192 │   10 │ 2137.2 │  186.6 │   9213.5 │    3 │
│  16384 │    1 │ 1651.9 │   66.1 │   1241.3 │    3 │
│  16384 │    2 │ 1736.1 │   94.6 │   2188.7 │    3 │
│  16384 │    5 │ 1857.8 │  133.6 │   5259.4 │    3 │
│  16384 │   10 │ 1900.3 │  162.4 │  10069.3 │    3 │
│  32768 │    1 │ 1339.1 │   54.9 │   1532.7 │    3 │
│  32768 │    2 │ 1529.0 │   70.0 │   2387.5 │    3 │
│  32768 │    5 │ 1584.3 │  119.0 │   6191.4 │    3 │
│  32768 │   10 │ 1639.4 │  132.3 │  11623.1 │    3 │
│  65535 │    1 │  828.7 │   38.0 │   2473.6 │    3 │
│  65535 │    2 │  865.7 │   63.4 │   4363.9 │    3 │
│  65535 │    5 │  975.5 │   86.6 │  10007.6 │    3 │
│  65535 │   10 │  863.8 │   67.8 │  20808.4 │    3 │
│ 100000 │    1 │  626.1 │   47.3 │   3273.8 │    3 │
│ 100000 │    2 │  673.8 │   55.1 │   5640.7 │    3 │
│ 100000 │    5 │  717.4 │   73.0 │  13588.8 │    3 │
│ 100000 │   10 │   68.5 │    4.9 │ 159905.0 │    3 │

vllm version 0.26.1rc1.dev371+g85ea44b46.d20260805
```

```bash
THe-Plague/Qwen3.6-35B-A3B-abliterated-NVFP4-MTP
--attention-backend FLASHINFER \
--moe-backend flashinfer_b12x \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN","moe_backend":"flashinfer_cutlass"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 13438145445 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24

* flashinfer_b12x uses +20gb more VRAM than alternatives, not worth it for a single spark as it means you can only run 1 model

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 4683.2 │   58.2 │   438.9 │    3 │
│     0 │    2 │ 6105.8 │   93.9 │   670.9 │    3 │
│     0 │    5 │ 6146.1 │  148.7 │  1584.1 │    3 │
│     0 │   10 │ 7296.3 │  282.3 │  2691.8 │    3 │
│  4096 │    1 │ 1852.5 │   45.0 │  1108.7 │    3 │
│  4096 │    2 │ 2214.2 │   93.4 │  1747.1 │    3 │
│  4096 │    5 │ 2391.7 │  133.0 │  4015.4 │    3 │
│  4096 │   10 │ 2496.2 │  189.6 │  7647.3 │    3 │
│  8192 │    1 │ 1563.3 │   29.5 │  1339.0 │    3 │
│  8192 │    2 │ 1971.2 │   77.8 │  1967.2 │    3 │
│  8192 │    5 │ 2159.4 │  107.7 │  4487.7 │    3 │
│  8192 │   10 │ 2272.5 │  204.9 │  8573.6 │    3 │
│ 16384 │    1 │ 1536.4 │   55.7 │  1336.2 │    3 │
│ 16384 │    2 │ 1771.9 │   77.1 │  2248.7 │    3 │
│ 16384 │    5 │ 1936.0 │  137.1 │  4886.3 │    3 │
│ 16384 │   10 │ 2035.9 │  157.0 │  9613.5 │    3 │
│ 32768 │    1 │ 1356.8 │   52.2 │  1512.3 │    3 │
│ 32768 │    2 │ 1485.6 │   70.7 │  2550.8 │    3 │
│ 32768 │    5 │ 1652.3 │  112.3 │  5898.5 │    3 │

vllm version 0.26.1rc1.dev371+g85ea44b46.d20260805
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--quantization modelopt \
--attention-backend triton_attn \
--moe-backend marlin \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn","moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 13438145445 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 8

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5007.4 │   95.3 │   431.5 │    3 │
│     0 │    2 │ 4931.8 │  140.5 │   794.8 │    3 │
│     0 │    5 │ 6472.0 │  196.0 │  1468.5 │    3 │
│     0 │   10 │ 3542.1 │  205.7 │  3008.2 │    3 │
│  4096 │    1 │ 1931.1 │   94.4 │  1062.3 │    3 │
│  4096 │    2 │ 2006.3 │  123.9 │  1866.2 │    3 │
│  4096 │    5 │ 2131.0 │  165.2 │  4571.1 │    3 │
│  4096 │   10 │ 1658.1 │  151.0 │  8292.1 │    3 │
│  8192 │    1 │ 1696.0 │   87.0 │  1209.4 │    3 │
│  8192 │    2 │ 1758.0 │  116.3 │  2132.6 │    3 │
│  8192 │    5 │ 1870.3 │  157.0 │  5211.3 │    3 │
│  8192 │   10 │ 1490.0 │  140.2 │  9380.7 │    3 │
│ 16384 │    1 │ 1411.0 │   79.6 │  1453.0 │    3 │
│ 16384 │    2 │ 1456.9 │  111.2 │  2580.4 │    3 │
│ 16384 │    5 │ 1545.1 │  137.9 │  6318.6 │    3 │
│ 16384 │   10 │ 1245.8 │  114.0 │ 11033.7 │    3 │
│ 32768 │    1 │ 1109.9 │   65.4 │  1847.0 │    3 │
│ 32768 │    2 │ 1143.8 │   89.4 │  3312.0 │    3 │

vllm version 0.23.1rc1.dev1302+ge765bbc97.d20260720
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--quantization modelopt \
--attention-backend triton_attn \
--moe-backend marlin \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn","moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 13438145445 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 16

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5583.3 │   90.2 │   370.3 │    3 │
│     0 │    2 │ 5379.8 │  139.8 │   781.3 │    3 │
│     0 │    5 │ 6407.3 │  199.8 │  1482.0 │    3 │
│     0 │   10 │ 6779.0 │  274.5 │  2909.0 │    3 │
│  4096 │    1 │ 1922.6 │   97.1 │  1066.8 │    3 │
│  4096 │    2 │ 2003.8 │  118.8 │  1870.6 │    3 │
│  4096 │    5 │ 2126.2 │  162.7 │  4581.0 │    3 │
│  4096 │   10 │ 2150.5 │  203.8 │  9076.7 │    3 │
│  8192 │    1 │ 1702.6 │   85.3 │  1204.9 │    3 │
│  8192 │    2 │ 1772.6 │  120.0 │  2115.5 │    3 │
│  8192 │    5 │ 1871.6 │  158.4 │  5208.7 │    3 │
│  8192 │   10 │ 1893.7 │  194.4 │ 10368.9 │    3 │
│ 16384 │    1 │ 1414.9 │   75.4 │  1449.0 │    3 │
│ 16384 │    2 │ 1465.2 │  104.7 │  2565.8 │    3 │
│ 16384 │    5 │ 1545.2 │  142.9 │  6318.4 │    3 │
│ 16384 │   10 │ 1560.6 │  138.4 │ 11836.2 │    3 │
│ 32768 │    1 │ 1125.8 │   57.7 │  1820.4 │    3 │
│ 32768 │    2 │ 1158.7 │   85.6 │  3267.0 │    3 │

vllm version 0.23.1rc1.dev1302+ge765bbc97.d20260720
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--quantization modelopt \
--attention-backend flashinfer \
--moe-backend marlin \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn","moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 13438145445 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24

               total        used        free      shared  buff/cache   available
Mem:           121Gi        57Gi        40Gi       215Mi        25Gi        63Gi
Swap:           15Gi       560Mi        15Gi

┃  depth ┃ conc ┃ pp t/s ┃ tg t/s ┃  ttfr ms ┃ runs ┃
│      0 │    1 │ 4972.8 │   69.2 │    415.0 │    3 │
│      0 │    2 │ 5828.8 │  134.3 │    614.8 │    3 │
│      0 │    5 │ 6542.7 │  193.5 │   1454.3 │    3 │
│      0 │   10 │ 6919.7 │  282.0 │   2850.1 │    3 │
│   4096 │    1 │ 2036.3 │   99.8 │   1007.7 │    3 │
│   4096 │    2 │ 2126.0 │  134.7 │   1764.0 │    3 │
│   4096 │    5 │ 2261.7 │  172.9 │   4310.9 │    3 │
│   4096 │   10 │ 2286.0 │  207.0 │   8547.0 │    3 │
│   8192 │    1 │ 1897.2 │  108.1 │   1081.9 │    3 │
│   8192 │    2 │ 1982.2 │  138.2 │   1893.4 │    3 │
│   8192 │    5 │ 2113.9 │  167.4 │   4612.9 │    3 │
│   8192 │   10 │ 2132.3 │  192.2 │   8941.1 │    3 │
│  16384 │    1 │ 1710.7 │  100.5 │   1199.4 │    3 │
│  16384 │    2 │ 1783.0 │  128.2 │   2108.4 │    3 │
│  16384 │    5 │ 1906.0 │  157.2 │   5119.4 │    3 │
│  16384 │   10 │ 1926.4 │  170.9 │   9771.7 │    3 │
│  32768 │    1 │ 1491.5 │   90.8 │   1375.3 │    3 │
│  32768 │    2 │ 1559.0 │  116.4 │   2426.4 │    3 │
│  32768 │    5 │ 1665.4 │  145.7 │   5874.4 │    3 │
│  32768 │   10 │ 1681.3 │  125.7 │  10865.4 │    3 │
│  65535 │    1 │  926.1 │   81.3 │   2213.7 │    3 │
│  65535 │    2 │  956.1 │   88.6 │   3922.1 │    3 │
│  65535 │    5 │ 1007.0 │   81.0 │   8745.2 │    3 │
│  65535 │   10 │ 1027.2 │   74.2 │  17843.6 │    3 │
│ 100000 │    1 │  675.4 │   76.0 │   3034.0 │    3 │
│ 100000 │    2 │  705.9 │   75.9 │   5306.2 │    3 │
│ 100000 │    5 │  736.6 │   62.3 │  11912.5 │    3 │
│ 100000 │   10 │   97.2 │    7.5 │ 115859.3 │    3 │

vllm version 0.23.1rc1.dev1302+ge765bbc97.d20260720
```

[benchmark_qwen3.6-35b-a3b_spark-arena-v1_tp1.csv](assets/benchmark_qwen3.6-35b-a3b_spark-arena-v1_tp1-20260804200759-bns0cq7.csv)

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--quantization modelopt \
--attention-backend flashinfer \
--moe-backend flashinfer_b12x \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn","moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 13438145445 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24

               total        used        free      shared  buff/cache   available
Mem:           121Gi        81Gi        16Gi       206Mi        24Gi        40Gi
Swap:           15Gi       567Mi        15Gi

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 6040.1 │  102.4 │   342.8 │    3 │
│     0 │    2 │ 4992.2 │  142.6 │   753.5 │    3 │
│     0 │    5 │ 7077.7 │  186.6 │  1332.6 │    3 │
│     0 │   10 │ 7547.2 │  268.6 │  2602.7 │    3 │
│  4096 │    1 │ 2118.8 │  105.2 │   968.8 │    3 │
│  4096 │    2 │ 2252.9 │  122.3 │  1652.3 │    3 │
│  4096 │    5 │ 2461.1 │  165.0 │  3937.6 │    3 │
│  4096 │   10 │ 2486.9 │  203.7 │  7814.7 │    3 │
│  8192 │    1 │ 1977.5 │  105.3 │  1037.1 │    3 │
│  8192 │    2 │ 2092.4 │  122.1 │  1783.1 │    3 │
│  8192 │    5 │ 2279.5 │  154.7 │  4250.7 │    3 │
│  8192 │   10 │ 2299.1 │  196.0 │  8450.5 │    3 │
│ 16384 │    1 │ 1764.1 │   98.0 │  1162.7 │    3 │
│ 16384 │    2 │ 1867.6 │  113.3 │  2003.4 │    3 │
│ 16384 │    5 │ 2029.4 │  151.4 │  4776.0 │    3 │
│ 16384 │   10 │ 2055.7 │  146.2 │  8814.5 │    3 │
│ 32768 │    1 │ 1539.0 │   92.0 │  1333.0 │    3 │
│ 32768 │    2 │ 1613.4 │  116.0 │  2339.8 │    3 │
│ 32768 │    5 │ 1712.9 │  120.9 │  5326.6 │    3 │
│ 32768 │   10 │ 1764.0 │  118.3 │ 10310.6 │    3 │
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--quantization modelopt \
--attention-backend flashinfer \
--moe-backend flashinfer_b12x \
--linear-backend flashinfer_b12x \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn","moe_backend":"triton"}' \

container: ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 13438145445 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24

               total        used        free      shared  buff/cache   available
Mem:           121Gi        82Gi        14Gi       202Mi        24Gi        38Gi
Swap:           15Gi       567Mi        15Gi

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 6127.9 │  102.3 │   338.1 │    3 │
│     0 │    2 │ 3857.6 │  136.9 │  1350.6 │    3 │
│     0 │    5 │ 6834.5 │  188.0 │  1389.1 │    3 │
│     0 │   10 │ 6839.6 │  265.9 │  2911.2 │    3 │
│  4096 │    1 │ 1856.7 │  106.0 │  1139.5 │    3 │
│  4096 │    2 │ 2209.0 │  129.7 │  1689.1 │    3 │
│  4096 │    5 │ 2352.2 │  164.9 │  4129.6 │    3 │
│  4096 │   10 │ 2430.2 │  202.0 │  8001.7 │    3 │
│  8192 │    1 │ 1961.4 │  104.8 │  1050.9 │    3 │
│  8192 │    2 │ 2043.3 │  128.5 │  1832.5 │    3 │
│  8192 │    5 │ 2217.2 │  160.8 │  4379.9 │    3 │
│  8192 │   10 │ 2205.8 │  192.6 │  8814.2 │    3 │
│ 16384 │    1 │ 1754.8 │   99.7 │  1171.9 │    3 │
│ 16384 │    2 │ 1836.5 │  115.3 │  2043.9 │    3 │
│ 16384 │    5 │ 2010.3 │  153.1 │  4828.9 │    3 │
│ 16384 │   10 │ 1990.7 │  131.5 │  9112.0 │    3 │
│ 32768 │    1 │ 1470.4 │   95.0 │  1396.6 │    3 │
│ 32768 │    2 │ 1623.3 │  113.9 │  2330.0 │    3 │
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--quantization modelopt \
--attention-backend flashinfer \
--moe-backend flashinfer_b12x \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"triton_attn","moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.4
  kv_cache_memory: 13438145445 # 2.01x
  max_model_len: 262144
  max_num_batched_tokens: 32768
  max_num_seqs: 24

               total        used        free      shared  buff/cache   available
Mem:           121Gi        81Gi        16Gi       206Mi        24Gi        40Gi
Swap:           15Gi       567Mi        15Gi

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 6040.1 │  102.4 │   342.8 │    3 │
│     0 │    2 │ 4992.2 │  142.6 │   753.5 │    3 │
│     0 │    5 │ 7077.7 │  186.6 │  1332.6 │    3 │
│     0 │   10 │ 7547.2 │  268.6 │  2602.7 │    3 │
│  4096 │    1 │ 2118.8 │  105.2 │   968.8 │    3 │
│  4096 │    2 │ 2252.9 │  122.3 │  1652.3 │    3 │
│  4096 │    5 │ 2461.1 │  165.0 │  3937.6 │    3 │
│  4096 │   10 │ 2486.9 │  203.7 │  7814.7 │    3 │
│  8192 │    1 │ 1977.5 │  105.3 │  1037.1 │    3 │
│  8192 │    2 │ 2092.4 │  122.1 │  1783.1 │    3 │
│  8192 │    5 │ 2279.5 │  154.7 │  4250.7 │    3 │
│  8192 │   10 │ 2299.1 │  196.0 │  8450.5 │    3 │
│ 16384 │    1 │ 1764.1 │   98.0 │  1162.7 │    3 │
│ 16384 │    2 │ 1867.6 │  113.3 │  2003.4 │    3 │
│ 16384 │    5 │ 2029.4 │  151.4 │  4776.0 │    3 │
│ 16384 │   10 │ 2055.7 │  146.2 │  8814.5 │    3 │
│ 32768 │    1 │ 1539.0 │   92.0 │  1333.0 │    3 │
│ 32768 │    2 │ 1613.4 │  116.0 │  2339.8 │    3 │
│ 32768 │    5 │ 1712.9 │  120.9 │  5326.6 │    3 │
│ 32768 │   10 │ 1764.0 │  118.3 │ 10310.6 │    3 │
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--moe-backend marlin \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \

container: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.3
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 32768
  max_num_seqs: 16

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5333.7 │  105.9 │   386.9 │    3 │
│     0 │    2 │ 4841.6 │  145.7 │   833.9 │    3 │
│     0 │    5 │ 5099.8 │  195.9 │  2333.5 │    3 │
│     0 │   10 │ 6945.5 │  383.0 │  2940.9 │    3 │
│  4096 │    1 │ 2014.4 │   92.2 │  1018.5 │    3 │
│  4096 │    2 │ 2122.6 │  155.0 │  1912.1 │    3 │
│  4096 │    5 │ 2283.1 │  227.2 │  4471.7 │    3 │
│  4096 │   10 │ 2309.7 │  314.4 │  8774.7 │    3 │
│  8192 │    1 │ 1866.5 │   94.6 │  1098.7 │    3 │
│  8192 │    2 │ 1985.7 │  158.9 │  2044.0 │    3 │
│  8192 │    5 │ 2142.7 │  229.8 │  4763.3 │    3 │
│  8192 │   10 │ 2169.2 │  328.8 │  9363.3 │    3 │
│ 16384 │    1 │ 1716.4 │   98.2 │  1194.8 │    3 │
│ 16384 │    2 │ 1805.4 │  151.2 │  2248.2 │    3 │
│ 16384 │    5 │ 1950.1 │  224.2 │  5232.8 │    3 │
│ 16384 │   10 │ 1970.2 │  182.2 │ 10027.2 │    3 │
│ 32768 │    1 │ 1509.8 │   98.1 │  1358.1 │    3 │
│ 32768 │    2 │ 1586.4 │  140.0 │  2559.8 │    3 │
│ 32768 │    5 │ 1684.6 │  173.1 │  5892.4 │    3 │

vllm version 0.26.1rc1.dev247+ge92dc7a9c.d20260802
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--moe-backend flashinfer_b12x \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
vars:
  gpu_memory_utilization: 0.6
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* flashinfer_b12x needs way more vram than marlin

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5858.2 │   98.1 │   351.5 │    3 │
│     0 │    2 │ 5221.3 │  129.6 │   761.4 │    3 │
│     0 │    5 │ 7087.4 │  223.4 │  1433.0 │    3 │
│     0 │   10 │ 7533.6 │  350.4 │  2711.7 │    3 │
│  4096 │    1 │ 2145.1 │  102.3 │   956.1 │    3 │
│  4096 │    2 │ 2288.1 │  152.7 │  1771.8 │    3 │
│  4096 │    5 │ 2474.4 │  220.9 │  4124.1 │    3 │
│  4096 │   10 │ 2514.2 │  300.6 │  8050.9 │    3 │
│  8192 │    1 │ 1995.2 │   94.8 │  1028.1 │    3 │
│  8192 │    2 │ 2105.0 │  155.2 │  1926.3 │    3 │
│  8192 │    5 │ 2309.6 │  233.5 │  4418.4 │    3 │
│  8192 │   10 │ 2344.3 │  318.8 │  8664.1 │    3 │
│ 16384 │    1 │ 1772.6 │  104.9 │  1157.2 │    3 │
│ 16384 │    2 │ 1886.9 │  150.5 │  2149.8 │    3 │
│ 16384 │    5 │ 2066.0 │  219.8 │  4933.5 │    3 │
│ 16384 │   10 │ 2099.0 │  174.2 │  9349.2 │    3 │
│ 32768 │    1 │ 1536.9 │   96.3 │  1334.4 │    3 │
│ 32768 │    2 │ 1640.5 │  150.4 │  2474.2 │    3 │
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--moe-backend flashinfer_b12x \
--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":16}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
vars:
  gpu_memory_utilization: 0.7
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* flashinfer_b12x needs way more vram than marlin
* dflash needs more vram than triton

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5393.9 │   61.3 │   382.3 │    3 │
│     0 │    2 │ 5604.6 │   97.7 │   697.2 │    3 │
│     0 │    5 │ 6872.4 │  159.8 │  1470.6 │    3 │
│     0 │   10 │ 2743.3 │  131.5 │  3939.4 │    3 │
│  4096 │    1 │ 1947.1 │   55.7 │  1053.7 │    3 │
│  4096 │    2 │ 2173.9 │  105.9 │  1850.8 │    3 │
│  4096 │    5 │ 1356.3 │   97.3 │  4199.9 │    3 │

(APIServer pid=103) INFO 08-03 10:01:01 [loggers.py:310] Engine 000: Avg prompt throughput: 2465.2 tokens/s, Avg generation throughput: 63.8 tokens/s, Running: 5 reqs, Waiting: 3 reqs, GPU KV cache usage: 98.0%, Prefix cache hit rate: 0.0%
(APIServer pid=103) INFO 08-03 10:01:01 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 3.27, Accepted throughput: 44.30 tokens/s, Drafted throughput: 311.98 tokens/s, Accepted: 443 tokens, Drafted: 3120 tokens, Per-position acceptance rate: 0.677, 0.446, 0.303, 0.190, 0.154, 0.123, 0.092, 0.067, 0.046, 0.041, 0.031, 0.021, 0.021, 0.021, 0.021, 0.021, Avg Draft acceptance rate: 14.2%
```

```bash
nvidia/Qwen3.6-35B-A3B-NVFP4
--moe-backend flashinfer_b12x \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
vars:
  gpu_memory_utilization: 0.6
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 32768
  max_num_seqs: 16

* flashinfer_b12x needs way more vram than marlin
* no speculation, tg quickly degrades with concurrence

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 6795.0 │   77.4 │   303.5 │    3 │
│     0 │    2 │ 6870.4 │  109.6 │   499.7 │    3 │
│     0 │    5 │ 7957.3 │  143.5 │  1157.0 │    3 │
│     0 │   10 │ 8243.9 │  191.4 │  2355.1 │    3 │
```

```bash
unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
--moe-backend marlin \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.3
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 32768
  max_num_seqs: 16

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5487.0 │   94.8 │   374.9 │    3 │
│     0 │    2 │ 4889.4 │  130.4 │   821.7 │    3 │
│     0 │    5 │ 6298.7 │  221.4 │  1613.9 │    3 │
│     0 │   10 │ 6008.6 │  341.8 │  3401.9 │    3 │
│  4096 │    1 │ 1984.5 │   92.3 │  1034.4 │    3 │
│  4096 │    2 │ 2084.0 │  143.9 │  1945.1 │    3 │
│  4096 │    5 │ 1958.2 │  220.2 │  5214.6 │    3 │
│  4096 │   10 │ 2002.1 │  317.6 │ 10131.8 │    3 │
│  8192 │    1 │ 1873.8 │   87.5 │  1094.0 │    3 │
│  8192 │    2 │ 1967.7 │  126.2 │  2060.6 │    3 │
│  8192 │    5 │ 1864.2 │  216.6 │  5476.9 │    3 │
│  8192 │   10 │ 1900.3 │  314.2 │ 10706.1 │    3 │
│ 16384 │    1 │ 1699.3 │   83.7 │  1207.2 │    3 │
│ 16384 │    2 │ 1789.6 │  140.2 │  2265.9 │    3 │
│ 16384 │    5 │ 1727.6 │  208.4 │  5906.0 │    3 │
│ 16384 │   10 │ 1763.2 │  243.4 │ 11431.6 │    3 │
│ 32768 │    1 │ 1498.2 │   91.9 │  1368.6 │    3 │
│ 32768 │    2 │ 1579.1 │  129.8 │  2568.5 │    3 │
│ 32768 │    5 │ 1581.1 │  190.1 │  6443.9 │    3 │
```

```bash
unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
--moe-backend flashinfer_b12x \
--linear-backend flashinfer_b12x \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \

container: ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest
env:
  CUTE_DSL_ARCH: sm_121a
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
vars:
  gpu_memory_utilization: 0.6
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 32768
  max_num_seqs: 16

┃ depth ┃ conc ┃ pp t/s ┃ tg t/s ┃ ttfr ms ┃ runs ┃
│     0 │    1 │ 5008.7 │   96.8 │   431.6 │    3 │
│     0 │    2 │ 4701.5 │  127.7 │  1014.0 │    3 │
│     0 │    5 │ 5287.4 │  174.2 │  2357.4 │    3 │
│     0 │   10 │ 6522.6 │  335.6 │  3132.0 │    3 │
│  4096 │    1 │ 2166.7 │   90.3 │   946.4 │    3 │
│  4096 │    2 │ 2274.3 │  137.6 │  1781.6 │    3 │
│  4096 │    5 │ 2148.5 │  204.8 │  4751.6 │    3 │
│  4096 │   10 │ 2185.0 │  296.4 │  9272.0 │    3 │
│  8192 │    1 │ 2020.0 │   88.6 │  1015.3 │    3 │
│  8192 │    2 │ 2129.4 │  142.2 │  1902.9 │    3 │
│  8192 │    5 │ 2010.3 │  201.1 │  5077.8 │    3 │
│  8192 │   10 │ 2068.0 │  321.6 │  9839.7 │    3 │
│ 16384 │    1 │ 1813.5 │   92.2 │  1130.9 │    3 │
```

```bash
unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
--moe-backend flashinfer_b12x \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \

container: eugr/spark-vllm:latest
env:
  CUTE_DSL_ARCH: sm_121a
vars:
  gpu_memory_utilization: 0.6
  max_model_len: 131072 # HALF CONTEXT!!!
  max_num_batched_tokens: 8192
  max_num_seqs: 8

┃  depth ┃ conc ┃          pp t/s ┃       tg t/s ┃ ttfr ms ┃
│      0 │    1 │  5543.3 ± 201.5 │  103.9 ± 4.7 │   372.3 │
│      0 │    2 │  5043.7 ± 210.0 │  124.4 ± 9.8 │   796.2 │
│      0 │    5 │  6399.9 ± 269.5 │  212.4 ± 5.4 │  1546.4 │
│      0 │   10 │   3497.7 ± 71.4 │  195.1 ± 9.8 │  2881.3 │
│   4096 │    1 │   2123.7 ± 17.5 │   88.4 ± 3.4 │   966.0 │
│   4096 │    2 │   2245.2 ± 13.1 │ 141.4 ± 12.2 │  1803.7 │
│   4096 │    5 │    2299.2 ± 1.2 │  119.6 ± 3.5 │  3514.3 │
│   4096 │   10 │    1764.3 ± 1.9 │  110.5 ± 0.4 │  6246.1 │
│   8192 │    1 │    1978.6 ± 2.5 │   87.9 ± 3.4 │  1037.1 │
│   8192 │    2 │    2091.7 ± 3.5 │  136.9 ± 0.9 │  1937.3 │
│   8192 │    5 │    2158.9 ± 0.8 │  129.7 ± 2.5 │  3848.2 │
│   8192 │   10 │   1662.5 ± 31.0 │  110.8 ± 2.5 │  6941.1 │
│  16384 │    1 │   1778.0 ± 19.2 │   89.9 ± 3.8 │  1153.8 │
│  16384 │    2 │    1878.0 ± 1.7 │  143.3 ± 1.2 │  2159.1 │
│  16384 │    5 │    1965.6 ± 1.7 │  124.3 ± 1.6 │  4250.5 │
│  16384 │   10 │    1530.3 ± 7.8 │  104.4 ± 0.7 │  7609.4 │
│  32768 │    1 │   1503.0 ± 34.8 │   79.0 ± 5.1 │  1365.2 │
│  32768 │    2 │    1631.3 ± 2.3 │  129.8 ± 6.4 │  2486.3 │
│  32768 │    5 │    1715.1 ± 2.8 │  126.0 ± 3.4 │  5249.9 │
│  32768 │   10 │   1352.8 ± 11.2 │   97.6 ± 0.7 │  8902.5 │
│  65535 │    1 │     970.5 ± 1.2 │   84.6 ± 1.7 │  2111.5 │
│  65535 │    2 │    1013.9 ± 1.5 │  121.2 ± 6.2 │  4010.5 │
│  65535 │    5 │    1065.4 ± 0.4 │   84.4 ± 1.5 │  7894.3 │
│  65535 │   10 │     899.6 ± 7.0 │   66.2 ± 0.7 │ 13756.2 │
│ 100000 │    1 │     714.1 ± 0.7 │   87.9 ± 5.6 │  2869.9 │
│ 100000 │    2 │     748.0 ± 3.7 │  113.5 ± 2.8 │  5439.1 │
│ 100000 │    5 │     782.8 ± 3.6 │   70.0 ± 1.5 │ 10740.4 │
│ 100000 │   10 │     672.2 ± 3.4 │   51.0 ± 0.6 │ 18555.2 │
```
