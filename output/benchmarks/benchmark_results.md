# Benchmark Results

*Last updated: 2025-11-12 22:01:32*

## Summary Table

| timestamp           |   vocab_size |   context_length |   d_model |   num_layers |   num_heads |   d_ff |   rope_theta |   batch_size | mode             | device   |   total_params_M |   mean_time_s |   median_time_s |   std_time_s |   min_time_s |   max_time_s |   steps_per_sec |   tokens_per_step |   tokens_per_sec |   gpu_allocated_GB |   gpu_reserved_GB |   gpu_max_allocated_GB |
|:--------------------|-------------:|-----------------:|----------:|-------------:|------------:|-------:|-------------:|-------------:|:-----------------|:---------|-----------------:|--------------:|----------------:|-------------:|-------------:|-------------:|----------------:|------------------:|-----------------:|-------------------:|------------------:|-----------------------:|
| 2025-11-12 21:56:33 |        10000 |              256 |       768 |           12 |          12 |   3072 |        10000 |            4 | forward_backward | cuda     |          128.625 |        0.1161 |          0.1149 |       0.0026 |       0.1132 |       0.1209 |          8.6109 |              1024 |          8817.52 |              2.188 |             3.496 |                  3.259 |
| 2025-11-12 21:59:09 |        10000 |              256 |      1024 |           24 |          16 |   4096 |        10000 |            4 | forward_backward | cuda     |          423.183 |        0.331  |          0.3307 |       0.0015 |       0.3294 |       0.3366 |          3.0216 |              1024 |          3094.11 |              6.796 |             9.305 |                  9.116 |
| 2025-11-12 22:01:32 |        10000 |              256 |      1280 |           36 |          20 |   5120 |        10000 |            4 | forward_backward | cuda     |          969.412 |        0.7653 |          0.765  |       0.0025 |       0.762  |       0.7691 |          1.3066 |              1024 |          1337.98 |             15.961 |            21.328 |                 19.428 |

## Notes

- `mean_time_s`: Average time per step in seconds
- `tokens_per_sec`: Throughput in tokens per second
- `total_params_M`: Total model parameters in millions
- `gpu_*_GB`: GPU memory usage in gigabytes
