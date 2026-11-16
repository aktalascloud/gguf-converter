# Share Safetensors FP8 E5M2 decoding

Use the same E5M2 decoder in media conversion and in Safetensors reference reads. Existing INT8 scales and W4A8 INT4 codebooks are consumed by the calibration reader when their metadata is present.

Base `bb96e56438adb325c07909c1d486e6995fea048b`.
