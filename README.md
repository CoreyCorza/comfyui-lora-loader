# Corza LoRA Loader (Clean)

A drop-in ComfyUI LoRA loader to reduce artifacts you often get from stacked LoRAs and from
distilled or turbo models (e.g. Krea‑2‑Turbo, Flux turbo etc).

With cleaning turned off it behaves **exactly** like the stock *Load LoRA* node, so you
can drop it in and A/B without changing anything else. The package also includes a
model-only inline cleaner for workflows that should keep using ComfyUI's normal LoRA
loaders.

## Why

Two things inside a LoRA file tend to cause artifacts, and few‑step (turbo) sampling
makes both worse because it never gets to average the noise out:

1. **Noise tail** — the low‑energy components of each layer's update are mostly training
   noise. They contribute little signal but plenty of high‑frequency crunch.
2. **Hot layers** — a handful of layers carry a much stronger update than the rest and
   shove activations off a distilled model's narrow manifold, giving you blockiness and
   fuzzy/aliased edges.

This node rewrites the LoRA's own low‑rank factors to address both — **before** ComfyUI
applies the LoRA — so it's model‑agnostic and needs no changes downstream.

## Install

**ComfyUI Manager:** search for `Corza LoRA Loader` and install.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CoreyCorza/comfyui-lora-loader
```
Restart ComfyUI. The nodes appear under **`corza/lora`**:

- **Corza LoRA Loader (Clean)** - a drop-in replacement for the stock LoRA loader.
- **Corza Clean Applied LoRAs** - a `MODEL -> MODEL` inline cleaner for LoRAs already
  applied upstream.

No extra dependencies — it only uses PyTorch and ComfyUI's own LoRA code.

## Usage

Wire it exactly like the stock LoRA loader (`model` in/out, optional `clip`). Leave the
cleanup controls at their defaults for identical behaviour, then dial them in:

| Input | Default | What it does |
|---|---|---|
| `strength_model` | 1.0 | Same as stock. |
| `strength_clip` | 1.0 | Same as stock. |
| `keep_energy` | 100 (off) | Per layer, keep only the strongest SVD components summing to this % of the update's energy; drop the rest as noise. Try **95**, then **90** if artifacts persist. |
| `max_rank` | 0 (off) | Hard cap on each layer's rank after the energy cut. |
| `tame_layers` | 0.0 (off) | Compress layers whose update is much stronger than the rest (above the LoRA's 90th percentile) back toward the pack. `0` = off, `1` = fully clamped. Try **0.5** for crunchy edges. |
| `star_rescale` | off | [STAR](https://arxiv.org/abs/2502.10339): after `keep_energy` trims a layer, boost the kept components so the layer's total strength (nuclear norm) matches the original. Lets you trim harder without weakening the LoRA's effect. Only active when `keep_energy` < 100. |

### Suggested starting point
For a style LoRA that's adding artifacts on a turbo model:
`keep_energy = 95`, `tame_layers = 0.5`. Compare against stock loading and back off if the
LoRA's effect gets too weak. When stacking multiple LoRAs, clean each one individually.

If trimming visibly weakens the LoRA, turn on `star_rescale` — it restores each trimmed
layer's strength, so you can push `keep_energy` down to 90 or below while keeping the
LoRA's full effect.

It logs a short report to the ComfyUI console per LoRA (rank saved, hottest layers) so you
can see what each one is doing.

### Inline model-path cleaner

Use **Corza Clean Applied LoRAs** when you want to keep the normal ComfyUI LoRA loaders.
Place it after the LoRA loaders you want cleaned:

`MODEL -> Load LoRA -> Load LoRA -> Corza Clean Applied LoRAs -> sampler`

The node only sees LoRA patches already present on its input `MODEL`, so downstream LoRA
loaders are ignored naturally:

`MODEL -> Load LoRA A -> Corza Clean Applied LoRAs -> Load LoRA B -> sampler`

In that graph, LoRA A is cleaned and LoRA B is not. This node is model-only, which fits
modern model families such as Krea 2 where the LoRA is applied to the diffusion model path
rather than CLIP.

The inline cleaner supports standard ComfyUI LoRA adapter patches. It deliberately skips
patches with DoRA scale, LoCon/Tucker mid weights, reshape metadata, or unknown adapter
types, matching the safety rules used by the drop-in loader.

## How it works

- **Energy trimming** uses the same math as kohya's `resize_lora` dynamic mode: each
  layer's update `Δ = up · down · (alpha / rank)` is decomposed with an exact SVD (via a
  cheap QR reduction on the low‑rank factors — seconds per LoRA, not minutes), and only the
  top singular components up to `keep_energy` are kept. The factors are rebuilt at the new
  rank with the scale baked in.
- **Taming** measures each layer's Frobenius norm, finds the 90th percentile across the
  LoRA, and scales down any layer above it toward that percentile by `tame_layers`.
- **STAR rescale** implements the rescale step from
  [STAR: Spectral Truncation and Rescale](https://arxiv.org/abs/2502.10339) (Lee et al.,
  2025): after truncation, the kept singular values are scaled by the ratio of original to
  kept nuclear norm, so removing the conflict-prone tail doesn't shrink the update.
- The transformed state dict is then handed to ComfyUI's own `load_lora_for_models`, so
  application is 100% standard.

**Compatibility:** architecture‑agnostic — anything ComfyUI can apply a LoRA to works
(SD/SDXL, Flux, Krea 2, etc.). Standard kohya and PEFT/diffusers key layouts are cleaned;
anything it doesn't recognise or can't safely refactor (DoRA, tucker/conv‑mid, reshape
metadata, unknown keys) is passed through untouched.

## Related

For resolving **conflicts between multiple different LoRAs** (TIES/DARE merging, per‑layer
conflict modes, autotuning), see
[ethanfel/ComfyUI-LoRA-Optimizer](https://github.com/ethanfel/ComfyUI-LoRA-Optimizer).
This node is complementary: it cleans a **single** LoRA well rather than merging many.

## License

MIT — see [LICENSE](LICENSE).
