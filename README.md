# Corza LoRA Loader (Clean)

A drop-in ComfyUI LoRA loader to reduce artifacts you often get from stacked LoRAs and from
distilled or turbo models (e.g. Krea‑2‑Turbo, Flux turbo etc).

With cleaning turned off it behaves **exactly** like the stock *Load LoRA* node, so you
can drop it in and A/B without changing anything else.

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
Restart ComfyUI. The node appears under **`corza/lora`** as **Corza LoRA Loader (Clean)**.

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

### Suggested starting point
For a style LoRA that's adding artifacts on a turbo model:
`keep_energy = 95`, `tame_layers = 0.5`. Compare against stock loading and back off if the
LoRA's effect gets too weak. When stacking multiple LoRAs, clean each one individually.

It logs a short report to the ComfyUI console per LoRA (rank saved, hottest layers) so you
can see what each one is doing.

## How it works

- **Energy trimming** uses the same math as kohya's `resize_lora` dynamic mode: each
  layer's update `Δ = up · down · (alpha / rank)` is decomposed with an exact SVD (via a
  cheap QR reduction on the low‑rank factors — seconds per LoRA, not minutes), and only the
  top singular components up to `keep_energy` are kept. The factors are rebuilt at the new
  rank with the scale baked in.
- **Taming** measures each layer's Frobenius norm, finds the 90th percentile across the
  LoRA, and scales down any layer above it toward that percentile by `tame_layers`.
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
