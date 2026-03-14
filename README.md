# AI Efficiency Metrics Cheatsheet

> Quick-reference guide covering memory, computation, performance, energy, and cost metrics for **AI Systems** — from CNNs and Transformers to TinyML and Diffusion models-based systems.

---

## Table of Contents

1. [Notation Reference](#notation-reference)
2. [Family I — Memory Metrics](#family-i--memory-metrics)
3. [Family II — Computation Metrics](#family-ii--computation-metrics)
4. [Family III — Performance & Latency Metrics](#family-iii--performance--latency-metrics)
5. [Family IV — Energy, Carbon & Cost Metrics](#family-iv--energy-carbon--cost-metrics)
6. [Layer-by-Layer Formulas](#layer-by-layer-formulas)
7. [Activation Formulas](#activation-formulas)
8. [Architecture-Specific Metrics](#architecture-specific-metrics)
9. [Hardware Constants Cheat Table](#hardware-constants-cheat-table)
10. [Quick Conversion Rules](#quick-conversion-rules)

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $c_i$ / $c_o$ | Input / output channels |
| $k_h$ / $k_w$ | Kernel height / width |
| $h_i$ / $w_i$ | Input feature map height / width |
| $h_o$ / $w_o$ | Output feature map height / width |
| $g$ | Number of groups (grouped convolution) |
| $n$ / $BS$ | Batch size |
| $L$ | Number of layers |
| $d$ / $d_{model}$ | Hidden / embedding dimension |
| $d_{head}$ | Per-head dimension ($d_{model} / \text{Heads}$) |
| $V$ | Vocabulary size |
| $N$ | Sequence length (tokens) |
| $b$ | Bit width (e.g., 32 for FP32, 16 for FP16, 8 for INT8) |
| $T$ | Number of diffusion sampling steps |

---

## Family I — Memory Metrics

| Metric | Formula | Unit | Notes |
|--------|---------|------|-------|
| **#Parameters ($W$)** | Sum of all weight tensor elements (see [layer table](#layer-by-layer-formulas)) | count | Bias typically ignored in estimates |
| **Model Size** | $W \times b$ | bits (convert to MB/GB) | Storage cost of weights on disk/flash |
| **#Activations (Total)** | $\sum_{\text{all layers}} \text{output activation elements}$ | count | Sum across all feature maps |
| **Peak #Activations** | $\approx \text{Input Activation} + \text{Output Activation}$ (bottleneck layer) | count | **Often the memory bottleneck** in inference |
| **Activation Memory** | $\text{\#Activations} \times b$ | bits → bytes | SRAM/GPU memory consumed |
| **KV Cache (LLMs)** | $BS \times L \times \text{Heads} \times d_{head} \times N \times 2 \times b$ | bits → bytes | Stores K & V for autoregressive decoding |

### Key Insight
> In CNN inference (especially TinyML), **Peak Activations** — not parameters — are the memory bottleneck. MobileNets reduce parameters but often **not** peak activation size.

---

## Family II — Computation Metrics

| Metric | Formula | Unit | Notes |
|--------|---------|------|-------|
| **MAC** | $a \leftarrow a + b \cdot c$ | count | One multiply + one accumulate |
| **FLOP** | $1 \text{ MAC} = 2 \text{ FLOPs}$ | count | Floating point operations |
| **Total FLOPs** | $\text{Total MACs} \times 2$ | count | |
| **OP** | Same as FLOP but for non-float (e.g., INT8) | count | Generalized operation count |
| **FLOPS / OPS** | Operations per second (hardware capability) | ops/s | FLOPS = FP; OPS = general |

### Transformer-Specific Computation

| Attention Type | Complexity (vs. sequence length $N$) |
|----------------|--------------------------------------|
| **Softmax Attention** | $O(N^2)$ |
| **Linear Attention** | $O(N)$ |

### Transformer FLOPs Heuristic (Decoder-Only)

$$
\text{FLOPs per token} \approx 6 \times L \times d^2
$$

$$
\text{Estimated params} \approx V \cdot d + L \times 12 \times d^2
$$

### Winograd Convolution (3×3)

Reduces cost from $9 \times C \times 4$ MACs → $16 \times C$ MACs for 4 outputs → **2.25× reduction**.

---

## Family III — Performance & Latency Metrics

### Latency

$$
\boxed{\text{Latency} \approx \max\left(T_{\text{compute}},\; T_{\text{memory}}\right)}
$$

| Component | Formula |
|-----------|---------|
| $T_{\text{compute}}$ | $\dfrac{\text{Total OPs in model}}{\text{OPS}_{\text{hardware}}}$ |
| $T_{\text{memory}}$ | $T_{\text{activations}} + T_{\text{weights}}$ |
| $T_{\text{activations}}$ | $\dfrac{\text{Input Act. Size} + \text{Output Act. Size}}{\text{Memory Bandwidth}}$ |
| $T_{\text{weights}}$ | $\dfrac{\text{Model Size}}{\text{Memory Bandwidth}}$ |

### Throughput

$$
\text{Throughput} = \frac{\text{Total Processed Units}}{\text{Total Time (s)}}
$$

Or simply: $\text{Throughput} \approx 1 / \text{Latency}$ (for single-stream).

### Tokens per Second (LLMs)

$$
\text{Tokens/s} = \frac{1}{T_{\text{per-token}}} = \frac{\text{OPS}_{\text{hardware}} \times \text{Utilization}}{\text{FLOPs per token}}
$$

---

## Family IV — Energy, Carbon & Cost Metrics

| Metric | Formula | Unit |
|--------|---------|------|
| **Device Power Draw** | $\text{TDP} \times \text{Utilization}$ | W |
| **Total Power (with PUE)** | $\text{Power Draw} \times \text{PUE}$ | W |
| **Energy per Inference** | $\dfrac{\text{Total Power} \times \text{Inference Time (s)}}{3600}$ | Wh |
| **OPS/W** | $\dfrac{\text{OPS/s}}{\text{Total Power (W)}}$ | ops/W |
| **OPS/Wh** | $\dfrac{\text{OPS per inference}}{\text{Energy per inference (Wh)}}$ | ops/Wh |
| **IPS/W** | $\dfrac{\text{Inferences/s}}{\text{Total Power (W)}}$ | inf/W |
| **TPS/W** | $\dfrac{\text{Tokens/s} \times BS}{\text{Total Power (W)}}$ | tok/W |
| **Carbon/Inference** | $\dfrac{\text{Energy (kWh)} \times \text{Grid Intensity (gCO₂/kWh)}}{1000}$ | kgCO₂ |
| **Cost/Inference** | $\text{Energy (kWh)} \times \text{Electricity Price (USD/kWh)}$ | USD |

### Key Insight
> **DRAM access ≈ 200× more energy** than a 32-bit arithmetic operation. Minimizing data movement is often more impactful than reducing FLOPs.

$$
\text{Energy} \propto \text{Data Movement} \rightarrow \text{More Memory References} \rightarrow \text{More Energy}
$$

---

## Layer-by-Layer Formulas

*Bias ignored. Batch size $n = 1$.*

| Layer Type | #Parameters | MACs |
|:-----------|:------------|:-----|
| **Fully-Connected (Linear)** | $c_o \cdot c_i$ | $c_o \cdot c_i$ |
| **Standard Convolution** | $c_o \cdot c_i \cdot k_h \cdot k_w$ | $c_o \cdot c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o$ |
| **Grouped Convolution** | $\dfrac{c_o \cdot c_i \cdot k_h \cdot k_w}{g}$ | $\dfrac{c_o \cdot c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o}{g}$ |
| **Depthwise Convolution** | $c_o \cdot k_h \cdot k_w$ | $c_o \cdot k_h \cdot k_w \cdot h_o \cdot w_o$ |
| **1×1 Convolution** | $c_o \cdot c_i$ | $c_o \cdot c_i \cdot h_o \cdot w_o$ |

> **Depthwise** = Grouped Conv where $g = c_i = c_o$. **1×1 Conv** = Standard Conv where $k_h = k_w = 1$.

---

## Activation Formulas

### Per-Layer Activation Sizes

| Layer Type | Input Activation | Output Activation |
|:-----------|:-----------------|:------------------|
| **CNN Layer** | $n \cdot c_i \cdot h_i \cdot w_i$ | $n \cdot c_o \cdot h_o \cdot w_o$ |
| **Linear Layer** | $n \cdot c_i$ | $n \cdot c_o$ |
| **Transformer Layer** | $BS \cdot N \cdot d_{model}$ | $BS \cdot N \cdot d_{model}$ |

### Peak vs. Total

| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| **Peak #Activations** | Max memory at any single point (HW constraint) | $\max_{\text{layer } l}\left(\text{Input}_l + \text{Output}_l\right)$ |
| **Total #Activations** | Sum of all feature maps across all layers | $\sum_{\text{all layers}} \text{Output}_l$ |
| **Activation Memory** | Byte cost of activations | $\text{\#Activations} \times b / 8$ bytes |

### Training Memory Note
> During **on-device training**, **all** intermediate activations from the forward pass must be stored for backpropagation — making memory $\gg$ inference-only. Sparse backprop can store only ~1/4 of activations.

---

## Architecture-Specific Metrics

### CNNs (Vision)

| Metric | Details |
|--------|---------|
| **Peak Activations** | Primary memory bottleneck; often larger than weights |
| **In-place Depthwise Conv** | Overwrites input buffer → memory = $\max(\text{In}, \text{Out})$ instead of $\text{In} + \text{Out}$ |
| **Winograd (3×3)** | 2.25× MAC reduction |

### Vision Transformers (ViT)

| Metric | Formula / Details |
|--------|-------------------|
| **Initial Token Count** | $\dfrac{H \times W}{\text{PatchSize}^2}$ |
| **Attention Cost** | $O(N^2)$ for softmax; $O(N)$ for linear attention |

### LLMs / Transformers

| Metric | Formula / Details |
|--------|-------------------|
| **KV Cache (MHA)** | $BS \times L \times \text{Heads} \times d_{head} \times N \times 2 \times b$ |
| **GQA Cache** | ~$8\times$ smaller than MHA |
| **MQA Cache** | ~$64\times$ smaller than MHA |
| **FLOPs/token** | $\approx 6 \times L \times d^2$ |
| **Estimated Params** | $\approx V \cdot d + 12 \cdot L \cdot d^2$ |
| **Effective Context (StreamingLLM)** | Uses "Attention Sinks" to handle context beyond window |

### Diffusion Models

| Metric | Formula / Details |
|--------|-------------------|
| **Total Latency** | $\text{Latency}_{\text{step}} \times T$ (sampling steps) |
| **FLOPs Scaling** | Linear with number of denoising steps $T$ |

### TinyML / Edge (MCU)

| Metric | Constraint |
|--------|------------|
| **Peak SRAM** | Must fit within ~320 KB |
| **Flash Usage** | = Model Size (weights only) |
| **SRAM Reuse (In-place DWConv)** | Output overwrites input buffer |

### Video Models

| Metric | Details |
|--------|---------|
| **TSM (Temporal Shift Module)** | 0 extra params, 0 extra MACs for temporal modeling; increases data movement cost |
| **Throughput** | Measured in Videos/s |

### Multimodal (VLM)

| Metric | Details |
|--------|---------|
| **Perceiver Resampler** | Maps variable visual features → fixed visual tokens (e.g., 5 tokens in Flamingo) |
| **Cross-Attention Overhead** | Cost between visual encoder and LLM backbone |

### 3D Point Clouds

| Metric | Details |
|--------|---------|
| **Sparsity** | Voxelized point clouds typically $<0.1\%$ dense |
| **PVCNN (Point-Voxel)** | Balances random memory access (point) vs. regular compute (voxel) |

---

## Hardware Constants Cheat Table

Define these before computing Family III & IV metrics:

| Parameter | Symbol | Example Value | Unit |
|-----------|--------|---------------|------|
| Processor Peak Performance | $\text{OPS}_{\text{hw}}$ | 100 | TOPS or TFLOPS |
| Memory Bandwidth | $\text{BW}_{\text{mem}}$ | 900 | GB/s |
| TDP (Thermal Design Power) | $\text{TDP}$ | 300 | W |
| Bit Width | $b$ | 16 | bits |
| Device Utilization | $U$ | 0.7 | fraction |
| PUE (Power Usage Effectiveness) | $\text{PUE}$ | 1.2 | ratio |
| Grid Carbon Intensity | — | 400 | gCO₂/kWh |
| Electricity Cost | — | 0.20 | USD/kWh |

### Common Energy Costs (Relative)

| Operation | Relative Energy |
|-----------|----------------|
| 32-bit INT Add | 1× (baseline) |
| 32-bit FP Multiply | ~4× |
| SRAM Read | ~6× |
| **DRAM Read** | **~200×** |

---

## Quick Conversion Rules

| From | To | Rule |
|------|----|------|
| MACs | FLOPs | $\times 2$ |
| MACs | OPs | $\times 2$ (non-float) |
| Parameters | Model Size (bytes) | $\times b / 8$ |
| Activations | Memory (bytes) | $\times b / 8$ |
| TFLOPS | FLOPS | $\times 10^{12}$ |
| Wh | kWh | $\div 1000$ |
| Latency (s) | Throughput (units/s) | $1 / \text{Latency}$ |
| Energy (kWh) → Carbon | $\text{kWh} \times \text{gCO}_2\text{/kWh}$ | gCO₂ |
| Energy (kWh) → Cost | $\text{kWh} \times \text{price}$ | USD |

---

## Compute-Bound vs. Memory-Bound Decision

$$
\text{Arithmetic Intensity} = \frac{\text{Total OPs}}{\text{Total Bytes Moved}}
$$

| Condition | Regime | Bottleneck |
|-----------|--------|------------|
| $\text{Arithmetic Intensity} > \dfrac{\text{OPS}_{\text{hw}}}{\text{BW}_{\text{mem}}}$ | **Memory-bound** | Data movement |
| $\text{Arithmetic Intensity} < \dfrac{\text{OPS}_{\text{hw}}}{\text{BW}_{\text{mem}}}$ | **Compute-bound** | Arithmetic |

> Use the **Roofline Model** to visualize where your workload sits.

---

## Distributed / Scaling Metrics

| Metric | Formula |
|--------|---------|
| **Scalability Ratio** | $\dfrac{\text{Throughput with } N \text{ GPUs}}{\text{Throughput with 1 GPU}}$ |
| **Communication Overhead** | $\dfrac{T_{\text{data transfer}}}{T_{\text{compute}} + T_{\text{data transfer}}}$ |

---

## Quantization Impact

| Precision | Bit Width | Model Size Reduction (vs FP32) | Typical Accuracy Impact |
|-----------|-----------|-------------------------------|------------------------|
| FP32 | 32 | 1× (baseline) | — |
| FP16 / BF16 | 16 | 2× | Minimal |
| INT8 | 8 | 4× | Small (< 1%) |
| INT4 | 4 | 8× | Moderate |

> Quantization error measured via **MSE** (Newton-Raphson clipping optimization).

---

## Efficiency Optimization Levers (Summary)

| Lever | Reduces | Metric Impact |
|-------|---------|---------------|
| **Pruning** | #Parameters, MACs | ↓ Model Size, ↓ FLOPs, ↓ Latency |
| **Quantization** | Bit Width | ↓ Model Size, ↓ Memory BW, ↑ OPS/W |
| **Knowledge Distillation** | Model complexity | ↓ Params while preserving accuracy |
| **Depthwise Separable Conv** | MACs, Params | ↓ FLOPs by ~8-9× vs standard conv |
| **GQA / MQA** | KV Cache size | ↓ Memory for LLM serving |
| **Linear Attention** | Attention cost | $O(N)$ vs $O(N^2)$ |
| **Winograd Transform** | Conv MACs | 2.25× reduction for 3×3 |
| **In-place DWConv** | Peak SRAM | Fits MCU memory constraints |
| **LoRA (PEFT)** | Training memory | Updates only rank-$r$ of weights |
| **Sparse Backprop** | Training activations | Stores ~1/4 of forward activations |
| **TSM (Video)** | Extra params/FLOPs | 0 overhead temporal modeling |

---

## References
- [MIT HAN Lab ](https://hanlab.mit.edu/)— [TinyML & Efficient AI lectures (Lec02-Basics)](https://hanlab.mit.edu/courses/2024-fall-65940)
