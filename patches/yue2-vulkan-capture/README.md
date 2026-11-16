# Capture YuE2 AR and NAR inputs on Vulkan

Name weights during GGUF upload and record activations at Vulkan matrix products. The collector accumulates single-row decode inputs and resolves full-width row views such as the semantic LM head. UMT5 lookup names are translated to the runtime's T5 projection names. The loader continues to use the original model and VAE from the profile. [Capture coverage](../../benchmarks/2026-11-17-capture-and-formats.json)
