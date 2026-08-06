# Nvidia GB10 (Spark) Recipes

The recipes in this repository follow 2 core principles:
- The best performance
- The best stability

> Most recipes in sites like https://spark-arena.com are very performant but are NOT stable for daily usage (crashes, low concurrency, missing core model parameters, chat templates, ...)

The most performant **and stable** recipes for running common models in your DGX Spark.

## Adapting the parameters

- Keep `gpu_memory_utilization` to just run 1 model in the whole spark.
- Keep `kv_cache_memory` to run multiple models in the same spark.

## Available recipes

- [Qwen3.6-35B-A3B](./qwen3.6-35b-a3b/README.md)
