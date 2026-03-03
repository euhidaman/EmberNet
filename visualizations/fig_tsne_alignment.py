"""
fig_tsne_alignment.py
=====================
t-SNE visualization of vision–text embedding alignment in the shared
768-dim space.  Called periodically during training to track how projected
image token embeddings and text token embeddings converge.

Each figure shows two point clouds:
  - Blue  = text token embeddings  (from decoder.embed_tokens)
  - Red   = image token embeddings (from vision_encoder → projector)

Over training steps the two clouds should progressively overlap as the
projector learns to map SigLIP features into the decoder's embedding space.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from visualizations.config import VIZ_CONFIG, apply_mpl_style, skip_no_data

apply_mpl_style()

MAX_TOKENS = 2000  # cap per modality to keep t-SNE fast


def collect_embeddings(model, batch: Optional[Dict] = None) -> Optional[Dict]:
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()

    text_embs = None
    img_embs = None

    with torch.no_grad():
        # --- Text embeddings ---
        if batch is not None and "input_ids" in batch:
            ids = batch["input_ids"].to(device)
        else:
            # fallback: embed a range of token IDs
            vocab = getattr(model.config, "vocab_size", 32002)
            ids = torch.arange(0, min(vocab, MAX_TOKENS), device=device).unsqueeze(0)

        safe_ids = ids.clamp(0, model.config.vocab_size - 1)
        text_embs = model.decoder.embed_tokens(safe_ids).float().squeeze(0).cpu().numpy()

        # De-duplicate identical rows & cap
        if text_embs.shape[0] > MAX_TOKENS:
            idx = np.random.default_rng(0).choice(text_embs.shape[0], MAX_TOKENS, replace=False)
            text_embs = text_embs[idx]

        # --- Image embeddings ---
        if batch is not None and "pixel_values" in batch:
            pv = batch["pixel_values"].to(device)
        else:
            img_size = 224
            try:
                img_size = getattr(model.config, "image_size", 224)
            except Exception:
                pass
            pv = torch.randn(2, 3, img_size, img_size, device=device, dtype=dtype)

        if pv.dim() == 5:
            pv = pv.view(-1, *pv.shape[2:])

        raw_img = model.vision_encoder(pv).float()
        # flatten [B, n_tok, dim] → [B*n_tok, dim]
        img_embs = raw_img.reshape(-1, raw_img.shape[-1]).cpu().numpy()
        if img_embs.shape[0] > MAX_TOKENS:
            idx = np.random.default_rng(1).choice(img_embs.shape[0], MAX_TOKENS, replace=False)
            img_embs = img_embs[idx]

    if text_embs is None or img_embs is None:
        return None

    return {"text": text_embs, "image": img_embs}


def _run_tsne(combined: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        # ultra-minimal fallback: PCA 2D
        mean = combined.mean(axis=0)
        centered = combined - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        top2 = eigvecs[:, -2:]
        return centered @ top2

    perp = min(perplexity, max(5.0, combined.shape[0] / 4.0))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                init="pca", learning_rate="auto", max_iter=800)
    return tsne.fit_transform(combined)


def plot_tsne(data: Dict, save_path: Path, step: Optional[int] = None) -> Path:
    text_emb = data["text"]
    img_emb = data["image"]

    n_text = text_emb.shape[0]
    n_img = img_emb.shape[0]
    combined = np.vstack([text_emb, img_emb]).astype(np.float32)

    # Remove any NaN/Inf rows
    mask = np.isfinite(combined).all(axis=1)
    combined = combined[mask]
    labels = np.array([0] * n_text + [1] * n_img)[mask]

    coords = _run_tsne(combined)

    text_mask = labels == 0
    img_mask = labels == 1

    fig, ax = plt.subplots(figsize=(8, 8), dpi=VIZ_CONFIG.get("dpi", 300))

    ax.scatter(coords[text_mask, 0], coords[text_mask, 1],
               c="#2176FF", s=12, alpha=0.55, label="Text tokens",
               edgecolors="none", rasterized=True)
    ax.scatter(coords[img_mask, 0], coords[img_mask, 1],
               c="#E03C31", s=18, alpha=0.55, label="Image tokens",
               marker="s", edgecolors="none", rasterized=True)

    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)

    step_str = f"Step {step}" if step is not None else "Snapshot"
    ax.set_title(f"Vision–Text Embedding Alignment ({step_str})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[tsne] Saved → {save_path}")
    return save_path


def generate(
    save_dir: Optional[Path] = None,
    model=None,
    batch: Optional[Dict] = None,
    step: Optional[int] = None,
) -> Optional[Path]:
    if save_dir is None:
        save_dir = Path("plots/paper_figures")

    if model is None:
        skip_no_data("fig_tsne_alignment")
        return None

    data = collect_embeddings(model, batch=batch)
    if data is None:
        print("[tsne] Could not collect embeddings, skipping.")
        return None

    out_dir = save_dir / "tsne_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    if step is not None:
        fname = f"tsne_alignment_step{step:06d}.png"
    else:
        fname = "tsne_alignment.png"

    return plot_tsne(data, out_dir / fname, step=step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE alignment visualization")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="plots/paper_figures")
    args = parser.parse_args()

    live_model = None
    if args.model_path:
        try:
            import torch
            from models.vlm import EmberNetVLM, EmberNetConfig
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
            cfg = EmberNetConfig()
            live_model = EmberNetVLM(cfg).to(device)
            live_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            live_model.eval()
            del ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[tsne] Model load failed: {e}")

    generate(save_dir=Path(args.output_dir), model=live_model)
