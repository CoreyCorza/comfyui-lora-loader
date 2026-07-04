"""Corza LoRA Loader (Clean) — LoRA loader with optional spectral cleanup.

Why: LoRAs (especially stacked, and especially on distilled/turbo models like
Krea-2-Turbo) often add blocky / crunchy artifacts with fuzzy edges. Two usual
culprits inside the LoRA file itself:
  1. noise — the low-energy tail of each layer's update is mostly training
     noise, and few-step turbo sampling amplifies it instead of averaging it out;
  2. hot layers — a handful of layers carry a far stronger update than the rest
     and push activations off the distilled model's narrow manifold.

This node is a drop-in LoRA loader (identical to the stock one with cleaning
off) that can rewrite the LoRA's low-rank factors on the fly:
  * keep_energy — per layer, SVD the update and keep only the strongest
    components adding up to this % of energy (same math as kohya's resize_lora
    dynamic rank; 90-98 is a sensible range).
  * max_rank    — hard cap on the per-layer rank after the energy cut.
  * tame_layers — compress outlier layers (Frobenius norm above the 90th
    percentile of this LoRA) back toward that percentile. 0 = off, 1 = fully
    clamped to the percentile.

The cleanup happens on the LoRA state dict *before* ComfyUI's own
load_lora_for_models call, so it is architecture-agnostic: anything ComfyUI can
apply a LoRA to (Krea 2, Flux, SDXL, ...) works unchanged. Exotic entries
(DoRA, tucker/conv mid matrices, reshape metadata, unknown keys) pass through
untouched. For cross-LoRA conflict resolution (TIES/DARE) see
ethanfel/ComfyUI-LoRA-Optimizer — this node deliberately stays complementary:
it cleans one LoRA well instead of merging many.
"""

import logging
import uuid

import torch

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io


# (up_suffix, down_suffix) naming schemes we know how to rebuild. Matches the
# common cases in comfy/weight_adapter/lora.py; anything else passes through.
_PAIR_SUFFIXES = [
    (".lora_up.weight", ".lora_down.weight"),   # kohya
    (".lora_B.weight", ".lora_A.weight"),       # peft / diffusers2
    ("_lora.up.weight", "_lora.down.weight"),   # diffusers
    (".lora.up.weight", ".lora.down.weight"),   # diffusers3
]


def _find_pairs(sd):
    """Yield (up_key, down_key, alpha_key_or_None, prefix) for cleanable pairs."""
    pairs = []
    for key in sd.keys():
        for up_sfx, down_sfx in _PAIR_SUFFIXES:
            if not key.endswith(up_sfx):
                continue
            prefix = key[: -len(up_sfx)]
            down_key = prefix + down_sfx
            if down_key not in sd:
                break
            # Skip variants whose patch math we'd break by refactoring:
            # tucker/conv mid matrix, DoRA magnitude, explicit reshape.
            if (prefix + ".lora_mid.weight") in sd:
                break
            if (prefix + ".dora_scale") in sd:
                break
            if (prefix + ".reshape_weight") in sd:
                break
            up, down = sd[key], sd[down_key]
            if up.ndim < 2 or down.ndim < 2:
                break
            # Conv LoRAs: up must be (out, r, 1, 1...) so a rank change is a
            # plain reshape; down (r, in, kh, kw) flattens per component.
            if any(d != 1 for d in up.shape[2:]):
                break
            alpha_key = prefix + ".alpha"
            pairs.append((key, down_key, alpha_key if alpha_key in sd else None, prefix))
            break
    return pairs


@torch.no_grad()
def _factorize(up, down, alpha):
    """Exact SVD of the low-rank delta via QR — cheap even for huge layers.

    delta = up @ down * (alpha / rank)  is rank <= r, so QR-reduce both factors
    and SVD only the tiny r x r core instead of the full out x in matrix.
    """
    rank = down.shape[0]
    scale = (float(alpha) / rank) if alpha is not None else 1.0
    u2 = up.reshape(up.shape[0], -1).float()      # out x r
    d2 = down.reshape(rank, -1).float()           # r x in
    q1, r1 = torch.linalg.qr(u2)                  # out x m1, m1 x r
    q2, r2 = torch.linalg.qr(d2.T)                # in x m2,  m2 x r
    core = (r1 @ r2.T) * scale                    # m1 x m2 (tiny)
    um, s, vmh = torch.linalg.svd(core)
    return {"q1": q1, "q2": q2, "um": um, "s": s, "vmh": vmh}


def _pick_rank(s, keep_energy, max_rank):
    energy = s * s
    total = float(energy.sum())
    if total <= 0.0:
        return 1
    cum = torch.cumsum(energy, dim=0) / total
    k = int(torch.searchsorted(cum, keep_energy / 100.0).item()) + 1
    k = min(k, s.shape[0])
    if max_rank > 0:
        k = min(k, max_rank)
    return max(k, 1)


@torch.no_grad()
def _rebuild(f, k, gain, up, down):
    """Refactor the truncated SVD back into up/down with the scale baked in.

    Splitting sqrt(S) across both factors keeps their magnitudes balanced.
    Caller rewrites .alpha to the new rank, so the loader's alpha/rank scale
    becomes exactly 1 and the baked-in scale is the only one applied.
    """
    s = f["s"][:k] * gain
    sq = torch.sqrt(s)
    new_up = (f["q1"] @ f["um"][:, :k]) * sq.unsqueeze(0)        # out x k
    new_down = sq.unsqueeze(1) * (f["vmh"][:k] @ f["q2"].T)      # k x in
    out_dim = up.shape[0]
    new_up = new_up.reshape(out_dim, k, *up.shape[2:])
    new_down = new_down.reshape(k, *down.shape[1:])
    return new_up.to(up.dtype).contiguous(), new_down.to(down.dtype).contiguous()


def _is_gate(prefix):
    """True for multiplicative gate projections (gated attention output gate,
    SwiGLU MLP gate, etc.) — e.g. Krea 2's blocks.N.attn.gate / blocks.N.mlp.gate,
    or generic gate_proj / to_gate names."""
    return any("gate" in seg.lower() for seg in prefix.split("."))


@torch.no_grad()
def _clean_lora(sd, keep_energy, max_rank, tame_layers, star_rescale, gate_strength, lora_name):
    pairs = _find_pairs(sd)
    if not pairs:
        logging.info(f"[Corza LoRA] {lora_name}: no cleanable up/down pairs found, loading as-is")
        return sd

    # Pass 1: factorize every pair and pick its kept rank.
    facts = []
    for up_key, down_key, alpha_key, prefix in pairs:
        up, down = sd[up_key], sd[down_key]
        alpha = sd[alpha_key].item() if alpha_key is not None else None
        f = _factorize(up, down, alpha)
        k = _pick_rank(f["s"], keep_energy, max_rank)
        # STAR rescale (arXiv:2502.10339): after truncating, boost the kept
        # components so the layer's nuclear norm matches the original — trimming
        # removes the conflict-prone tail without weakening the LoRA's effect.
        gain = 1.0
        if star_rescale and k < f["s"].shape[0]:
            nuc_all = float(f["s"].sum())
            nuc_kept = float(f["s"][:k].sum())
            if nuc_kept > 0.0:
                gain = nuc_all / nuc_kept
        # Gate scaling: gates are multiplicative sigmoid controls, so a LoRA delta
        # there has outsized nonlinear effect (and stacked LoRAs' gate edits compound
        # rather than average) — a common artifact / "loras fighting" source that
        # norm-based taming misses. Scale their update down by gate_strength.
        is_gate = _is_gate(prefix)
        if is_gate:
            gain *= gate_strength
        norm = float(torch.sqrt((f["s"][:k] ** 2).sum())) * gain
        facts.append({"up_key": up_key, "down_key": down_key, "alpha_key": alpha_key,
                      "prefix": prefix, "f": f, "k": k, "norm": norm, "gain": gain,
                      "star": (gain if star_rescale and k < f["s"].shape[0] else None),
                      "is_gate": is_gate})

    # Pass 2: tame outlier layers — compress norms above the 90th percentile
    # back toward it. tame_layers interpolates between untouched (0) and fully
    # clamped to the percentile (1). Runs on post-rescale norms.
    if tame_layers > 0.0 and len(facts) >= 4:
        norms = torch.tensor([x["norm"] for x in facts])
        q = float(torch.quantile(norms, 0.90))
        if q > 0.0:
            for x in facts:
                if x["norm"] > q:
                    target = q + (x["norm"] - q) * (1.0 - tame_layers)
                    x["gain"] *= target / x["norm"]
                    x["tamed"] = True

    # Pass 3: rebuild the state dict (leaving genuinely-untouched layers as-is).
    out = dict(sd)
    rank_in, rank_out, tamed, gated = 0, 0, 0, 0
    for x in facts:
        up, down = sd[x["up_key"]], sd[x["down_key"]]
        rank_in += down.shape[0]
        # nothing to do: full rank kept and no gain change → leave original tensors
        if x["k"] == down.shape[0] and abs(x["gain"] - 1.0) < 1e-9:
            rank_out += x["k"]
            continue
        new_up, new_down = _rebuild(x["f"], x["k"], x["gain"], up, down)
        out[x["up_key"]] = new_up
        out[x["down_key"]] = new_down
        if x["alpha_key"] is not None:
            # scale is baked into the factors → make alpha/rank come out as 1
            out[x["alpha_key"]] = torch.tensor(float(x["k"]))
        rank_out += x["k"]
        if x.get("tamed"):
            tamed += 1
        if x["is_gate"] and gate_strength != 1.0:
            gated += 1

    hottest = sorted(facts, key=lambda x: x["norm"], reverse=True)[:5]
    hot_txt = ", ".join(f"{x['prefix'].split('.')[-1] or x['prefix']}={x['norm']:.3f}" for x in hottest)
    star_txt = ""
    if star_rescale:
        gains = [x["star"] for x in facts if x["star"] is not None]
        if gains:
            star_txt = f", STAR-rescaled {len(gains)} layers (avg x{sum(gains) / len(gains):.2f})"
    gate_txt = ""
    n_gate = sum(1 for x in facts if x["is_gate"])
    if n_gate:
        gate_txt = f", {n_gate} gate layers"
        if gate_strength != 1.0:
            gate_txt += f" scaled x{gate_strength:.2f}"
    logging.info(
        f"[Corza LoRA] {lora_name}: cleaned {len(facts)} layers, "
        f"total rank {rank_in} -> {rank_out}, tamed {tamed} hot layers{star_txt}{gate_txt} | hottest: {hot_txt}"
    )
    return out


def _adapter_alpha_value(alpha):
    if alpha is None:
        return None
    if hasattr(alpha, "item"):
        return alpha.item()
    return alpha


def _cleanable_lora_adapter(adapter):
    if getattr(adapter, "name", None) != "lora":
        return None
    weights = getattr(adapter, "weights", None)
    if not isinstance(weights, (list, tuple)) or len(weights) < 6:
        return None

    up, down, alpha, mid, dora_scale, reshape = weights[:6]
    if mid is not None or dora_scale is not None or reshape is not None:
        return None
    if not torch.is_tensor(up) or not torch.is_tensor(down):
        return None
    if up.ndim < 2 or down.ndim < 2:
        return None
    if any(d != 1 for d in up.shape[2:]):
        return None
    return up, down, _adapter_alpha_value(alpha)


@torch.no_grad()
def _clean_model_lora_patches(model, keep_energy, max_rank, tame_layers, star_rescale, gate_strength):
    facts = []
    for patch_key, patch_list in getattr(model, "patches", {}).items():
        for patch_index, patch in enumerate(patch_list):
            if len(patch) < 2:
                continue
            adapter = patch[1]
            cleanable = _cleanable_lora_adapter(adapter)
            if cleanable is None:
                continue

            up, down, alpha = cleanable
            f = _factorize(up, down, alpha)
            k = _pick_rank(f["s"], keep_energy, max_rank)
            gain = 1.0
            if star_rescale and k < f["s"].shape[0]:
                nuc_all = float(f["s"].sum())
                nuc_kept = float(f["s"][:k].sum())
                if nuc_kept > 0.0:
                    gain = nuc_all / nuc_kept
            # gates: multiplicative sigmoid controls, outsized/compounding effect
            is_gate = _is_gate(patch_key)
            if is_gate:
                gain *= gate_strength
            norm = float(torch.sqrt((f["s"][:k] ** 2).sum())) * gain
            facts.append({
                "patch_key": patch_key,
                "patch_index": patch_index,
                "adapter": adapter,
                "up": up,
                "down": down,
                "alpha": alpha,
                "f": f,
                "k": k,
                "norm": norm,
                "gain": gain,
                "star": (gain if star_rescale and k < f["s"].shape[0] else None),
                "is_gate": is_gate,
            })

    if not facts:
        logging.info("[Corza LoRA] applied patch cleaner: no cleanable LoRA patches found")
        return model

    if tame_layers > 0.0 and len(facts) >= 4:
        norms = torch.tensor([x["norm"] for x in facts])
        q = float(torch.quantile(norms, 0.90))
        if q > 0.0:
            for x in facts:
                if x["norm"] > q:
                    target = q + (x["norm"] - q) * (1.0 - tame_layers)
                    x["gain"] *= target / x["norm"]
                    x["tamed"] = True

    out_model = model.clone()
    rank_in, rank_out, tamed = 0, 0, 0
    for x in facts:
        rank_in += x["down"].shape[0]
        rank_out += x["k"]
        # untouched (full rank, no gain change) → leave the original patch as-is
        if x["k"] == x["down"].shape[0] and abs(x["gain"] - 1.0) < 1e-9:
            continue
        new_up, new_down = _rebuild(x["f"], x["k"], x["gain"], x["up"], x["down"])
        new_alpha = float(x["k"]) if x["alpha"] is not None else None
        new_weights = (new_up, new_down, new_alpha, None, None, None)
        loaded_keys = getattr(x["adapter"], "loaded_keys", set())
        new_adapter = type(x["adapter"])(loaded_keys, new_weights)

        patch_list = out_model.patches[x["patch_key"]]
        patch = patch_list[x["patch_index"]]
        patch_list[x["patch_index"]] = (patch[0], new_adapter, *patch[2:])

        if x.get("tamed"):
            tamed += 1

    out_model.patches_uuid = uuid.uuid4()
    hottest = sorted(facts, key=lambda x: x["norm"], reverse=True)[:5]
    hot_txt = ", ".join(f"{x['patch_key']}[{x['patch_index']}]={x['norm']:.3f}" for x in hottest)
    star_txt = ""
    if star_rescale:
        gains = [x["star"] for x in facts if x["star"] is not None]
        if gains:
            star_txt = f", STAR-rescaled {len(gains)} patches (avg x{sum(gains) / len(gains):.2f})"
    gate_txt = ""
    n_gate = sum(1 for x in facts if x["is_gate"])
    if n_gate:
        gate_txt = f", {n_gate} gate patches"
        if gate_strength != 1.0:
            gate_txt += f" scaled x{gate_strength:.2f}"
    logging.info(
        f"[Corza LoRA] applied patch cleaner: cleaned {len(facts)} patches, "
        f"total rank {rank_in} -> {rank_out}, tamed {tamed} hot patches{star_txt}{gate_txt} | hottest: {hot_txt}"
    )
    return out_model


class CorzaLoRAClean(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CorzaLoRAClean",
            display_name="Corza LoRA Loader (Clean)",
            category="corza/lora",
            description="LoRA loader with optional spectral cleanup: drops the noisy "
                        "low-energy tail of each layer (kohya resize math) and tames "
                        "outlier layers. Helps with blocky/crunchy artifacts, "
                        "especially on distilled/turbo models and stacked LoRAs. "
                        "With cleaning off it behaves exactly like the stock loader.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", optional=True),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01),
                io.Float.Input("strength_clip", default=1.0, min=-100.0, max=100.0, step=0.01),
                io.Float.Input(
                    "keep_energy", default=100.0, min=50.0, max=100.0, step=0.5,
                    tooltip="Per layer, keep only the strongest SVD components adding up to "
                            "this % of the update's energy; the rest is mostly training "
                            "noise. 100 = off. Try 95, then 90 if artifacts persist.",
                ),
                io.Int.Input(
                    "max_rank", default=0, min=0, max=1024,
                    tooltip="Hard cap on each layer's rank after the energy cut. 0 = off.",
                ),
                io.Float.Input(
                    "tame_layers", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Compress layers whose update is much stronger than the rest "
                            "(above the LoRA's 90th percentile) back toward the pack. "
                            "0 = off, 1 = fully clamped. Try 0.5 for crunchy edges.",
                ),
                io.Boolean.Input(
                    "star_rescale", default=False,
                    tooltip="STAR (arXiv:2502.10339): after keep_energy trims a layer, boost "
                            "the kept components so the layer's total strength matches the "
                            "original — trims the conflict-prone tail without weakening the "
                            "LoRA. Only does something when keep_energy < 100.",
                ),
                io.Float.Input(
                    "gate_strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="How much of the LoRA's effect reaches 'gate' layers only (Krea 2's "
                            "gated attention + SwiGLU gates, etc.). Gates are multiplicative "
                            "sigmoid controls, so LoRA edits there have outsized, compounding "
                            "effect — a big source of artifacts and of stacked LoRAs fighting, "
                            "which the norm-based tame_layers misses. 1 = full effect (default), "
                            "0 = strip the LoRA from gates (they stay at base). Try 0.5 if "
                            "stacked LoRAs deform. No effect on LoRAs without gate layers.",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Clip.Output("clip"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        lora_name: str,
        strength_model: float = 1.0,
        strength_clip: float = 1.0,
        keep_energy: float = 100.0,
        max_rank: int = 0,
        tame_layers: float = 0.0,
        star_rescale: bool = False,
        gate_strength: float = 1.0,
        clip=None,
    ) -> io.NodeOutput:
        if strength_model == 0 and (clip is None or strength_clip == 0):
            return io.NodeOutput(model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

        cleaning = keep_energy < 100.0 or max_rank > 0 or tame_layers > 0.0 or gate_strength != 1.0
        if cleaning:
            sd = _clean_lora(sd, keep_energy, max_rank, tame_layers, star_rescale, gate_strength, lora_name)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, sd, strength_model, strength_clip
        )
        return io.NodeOutput(model_lora, clip_lora)


class CorzaCleanAppliedLoRAs(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CorzaCleanAppliedLoRAs",
            display_name="Corza Clean Applied LoRAs",
            category="corza/lora",
            description="Cleans standard LoRA patches already applied to the incoming "
                        "MODEL. Place it after upstream LoRA loaders; downstream LoRA "
                        "loaders remain untouched.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "keep_energy", default=100.0, min=50.0, max=100.0, step=0.5,
                    tooltip="Per applied LoRA patch, keep only the strongest SVD "
                            "components adding up to this % of the update's energy. "
                            "100 = off.",
                ),
                io.Int.Input(
                    "max_rank", default=0, min=0, max=1024,
                    tooltip="Hard cap on each applied LoRA patch's rank after the "
                            "energy cut. 0 = off.",
                ),
                io.Float.Input(
                    "tame_layers", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Compress applied LoRA patches whose update is much "
                            "stronger than the rest of the upstream LoRA stack. "
                            "0 = off, 1 = fully clamped.",
                ),
                io.Boolean.Input(
                    "star_rescale", default=False,
                    tooltip="STAR rescale after truncation. Only does something when "
                            "keep_energy trims a patch or max_rank caps it.",
                ),
                io.Float.Input(
                    "gate_strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="How much of the applied LoRA patches' effect reaches 'gate' layers "
                            "only (Krea 2 gated-attention + SwiGLU gates, etc.). Gates are "
                            "multiplicative sigmoid controls whose LoRA edits have outsized, "
                            "compounding effect — a big source of artifacts and of stacked "
                            "LoRAs fighting. 1 = full effect (default), 0 = strip the LoRA from "
                            "gates (they stay at base). Try 0.5 if stacked LoRAs deform.",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        keep_energy: float = 100.0,
        max_rank: int = 0,
        tame_layers: float = 0.0,
        star_rescale: bool = False,
        gate_strength: float = 1.0,
    ) -> io.NodeOutput:
        cleaning = keep_energy < 100.0 or max_rank > 0 or tame_layers > 0.0 or gate_strength != 1.0
        if not cleaning:
            return io.NodeOutput(model)
        model_lora = _clean_model_lora_patches(
            model,
            keep_energy=keep_energy,
            max_rank=max_rank,
            tame_layers=tame_layers,
            star_rescale=star_rescale,
            gate_strength=gate_strength,
        )
        return io.NodeOutput(model_lora)


NODE_CLASS_MAPPINGS = {
    "CorzaLoRAClean": CorzaLoRAClean,
    "CorzaCleanAppliedLoRAs": CorzaCleanAppliedLoRAs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CorzaLoRAClean": "Corza LoRA Loader (Clean)",
    "CorzaCleanAppliedLoRAs": "Corza Clean Applied LoRAs",
}
