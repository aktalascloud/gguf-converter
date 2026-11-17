# Support an F32 reference reconstructed from the current GGUF

An explicit `reference: null` selects streamed F32 reconstruction for the ACTCAL round trip. The original quant already gives zero loss against this reference. The converter therefore publishes an exact copy with cancellation and source identity checks. The shared GUI wiring also exposes MACaQ-R for the corresponding native quant formats. Apply after the MACaQ-R Vulkan solver patch.
