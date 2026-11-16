# Calibrate K-quant superblock scales on Vulkan

Decode the existing packed K blocks into one or two basis vectors. Fit the outer FP16 scale and, where present, the minimum scale by activation-weighted least squares. Select output rows on a separate subset of the capture. Packed codes, subscales, tensor types and byte lengths are preserved.

The format fixtures cover Q2_K, Q3_K, Q4_K, Q5_K and Q6_K on Polaris. [Portable checks](../../benchmarks/2026-11-17-capture-and-formats.json)
