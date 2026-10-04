---
library_name: sglang
base_model: LiquidAI/LFM2.5-8B-A1B
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- speculative-decoding
- dspark
- lfm2
- lfm2_moe
- draft-model
---

<div align="center">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png" 
    alt="Liquid AI" 
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> • 
    <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> • 
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a> • 
    <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
  </div>
</div>

# LFM2.5-8B-A1B-DSpark

**LFM2.5-DSpark** is a family of speculative-decoding draft models that adapt DSpark for the LFM2.5 architecture. 
They allow LFM2.5 models to run faster without degrading quality.

This is a drafter for **[`LiquidAI/LFM2.5-8B-A1B`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)**.
In SGLang, decoding runs about 2.6× faster. It also runs on-device on Apple silicon through the Metal backend.

Find more information about LFM2.5-DSpark in our [blog post](https://www.liquid.ai/blog/lfm2.5-dspark).

## 🗒️ Model Details

LFM2.5-8B-A1B-DSpark is a DSpark speculative-decoding draft model with the following features:

- **Target model**: [`LiquidAI/LFM2.5-8B-A1B`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- **Draft parameters**: **327.7M** (BF16)
- **Backbone**: 5 full attention layers, `hidden_size=2048`, `intermediate_size=6144` with SiLU/SwiGLU, GQA with `num_attention_heads=32` / `num_key_value_heads=8`, `head_dim=64`
- **Extra heads**: Markov head (rank 256) + confidence head
- **Block size**: 9
- **Vocabulary**: 128,000

Other models in the LFM2.5-DSpark family:

| Drafter | Target |
|---|---|
| [LFM2.5-1.2B-Instruct-DSpark](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-DSpark) | [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| [LFM2.5-8B-A1B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark) | [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) |
| [LFM2.5-2.6B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark) | [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |

## 📊 Performance

### Benchmarks

Speculative decoding is **exact**: the target verifies every proposed token, so the generated
text is what the target would have produced on its own. See [`LiquidAI/LFM2.5-8B-A1B`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) for performance benchmarks.

### Acceptance

Mean accepted tokens per decoding step, by benchmark (1×H100, batch size 1, greedy decoding).
Higher means more of the draft's proposed block is accepted per target forward pass, so decoding
is faster (at block size 9, the ceiling is 10).

| Benchmark | Accepted tokens / step |
|---|---:|
| MATH-500 | 8.02 |
| GSM8K | 3.91 |
| HumanEval | 7.48 |
| MBPP | 7.63 |
| MT-Bench | 8.99 |
| **Mean** | **7.21** |

### On-device and GPU Inference

| Dataset | Acceptance (of 10\) | Speedup on H100 | Speedup on M4 Max |
| :---- | :---- | :---- | :---- |
| MATH500 | 8.27 | **3.18x**<br/>428 → 1362 tok/s | **1.21x**<br/>93 → 112 tok/s |
| HumanEval | 7.02 | **2.58x**<br/>426 → 1100 tok/s | **1.12x**<br/>91 → 101 tok/s |
| MBPP | 6.93 | **2.64x**<br/>426 → 1122 tok/s | **1.09x**<br/>89 → 97 tok/s |
| GSM8K | 4.02 | **1.29x**<br/>385 → 496 tok/s | **1.44x**<br/>90 → 129 tok/s |
| MT-Bench | 8.52 | **3.02x**<br/>426 → 1288 tok/s | **1.04x**<br/>87 → 90 tok/s |
| Mean | 6.95 | **2.54x**<br/>418 → 1074 tok/s | **1.18x**<br/>90 → 106 tok/s |

## 🏃 How to run (SGLang)

Requires a build of SGLang with DSpark support for LFM2 / LFM2-MoE targets
([PR #31041](https://github.com/sgl-project/sglang/pull/31041)). Launch the target with the draft
attached:

```bash
python -m sglang.launch_server \
  --model-path LiquidAI/LFM2.5-8B-A1B \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path LiquidAI/LFM2.5-8B-A1B-DSpark \
  --speculative-draft-attention-backend flashinfer \
  --disable-radix-cache --mem-fraction-static 0.75 --port 30000
```

Then query the OpenAI-compatible endpoint at `http://localhost:30000/v1`. The block size is read
from the draft's `config.json`; the baseline is the same command without the three
`--speculative-*` flags.

## 📬 Contact

- Got questions or want to connect? [Join our Discord community](https://discord.com/invite/liquid-ai)
- If you are interested in custom solutions with edge deployment, please contact [our sales team](https://www.liquid.ai/contact).

## Citation

```bibtex
@article{liquidAI202626B,
  author  = {Liquid AI},
  title   = {LFM2.5-2.6B: Agents Everywhere},
  journal = {Liquid AI Blog},
  year    = {2026},
  note    = {www.liquid.ai/blog/lfm2-5-2-6b},
}
```

```bibtex
@article{liquidAI2026dspark,
  author = {Liquid AI},
  title = {LFM2.5-DSpark: Up to 3.2x Faster Inference from H100 to MacBook},
  journal = {Liquid AI Blog},
  year = {2026},
  note = {www.liquid.ai/blog/lfm2.5-dspark},
}
```
