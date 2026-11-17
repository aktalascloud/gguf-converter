# Refine native GGUF codes on Vulkan

Port the fixed-scale greedy MACaQ-R update to Vulkan using block-diagonal 32-column Hessians. Native code layouts preserve Q4_0, Q8_0, Q4_K, Q5_K and Q6_K scales and sizes. Independent activation rows select replacements. Source layout and algorithm attribution are included with the Apache notice. [MACaQ source](https://github.com/huawei-csl/MACaQ/tree/3810efca0b9e819c8eb51addd93c3c87225b7476/llamacpp/refine)
