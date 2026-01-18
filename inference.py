from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from groundingdino.util.inference import load_model, load_image, predict, annotate  # type: ignore


# =========================================================
# 1) HARD FORCE CPU (prevent any CUDA usage)
# =========================================================
DEVICE = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

torch.set_grad_enabled(False)
torch.set_num_threads(1)

# make CUDA look unavailable
torch.cuda.is_available = lambda: False  # type: ignore[attr-defined]
torch.cuda.device_count = lambda: 0      # type: ignore[attr-defined]
try:
    torch.backends.cudnn.enabled = False  # type: ignore[attr-defined]
except Exception:
    pass


# Redirect any ".to('cuda')" or ".cuda()" to CPU (safety net)
_orig_tensor_to = torch.Tensor.to
_orig_module_to = nn.Module.to

def _is_cuda(dev) -> bool:
    return "cuda" in str(dev).lower()

def _patched_tensor_to(self, *args, **kwargs):
    if args and _is_cuda(args[0]):
        new_args = list(args)
        new_args[0] = torch.device("cpu")
        return _orig_tensor_to(self, *new_args, **kwargs)
    if "device" in kwargs and _is_cuda(kwargs["device"]):
        kwargs["device"] = torch.device("cpu")
    return _orig_tensor_to(self, *args, **kwargs)

def _patched_module_to(self, *args, **kwargs):
    if args and _is_cuda(args[0]):
        new_args = list(args)
        new_args[0] = torch.device("cpu")
        return _orig_module_to(self, *new_args, **kwargs)
    if "device" in kwargs and _is_cuda(kwargs["device"]):
        kwargs["device"] = torch.device("cpu")
    return _orig_module_to(self, *args, **kwargs)

torch.Tensor.to = _patched_tensor_to  # type: ignore[assignment]
nn.Module.to = _patched_module_to     # type: ignore[assignment]

setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
setattr(nn.Module, "cuda", lambda self, *a, **k: self)


# =========================================================
# 2) AUTO-FIND CONFIG + WEIGHTS (no path problems)
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG_CANDIDATES = [
    PROJECT_ROOT / "GroundingDINO_SwinT_OGC.py",
    PROJECT_ROOT / "weights" / "GroundingDINO_SwinT_OGC.py",
    PROJECT_ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
    PROJECT_ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py".replace("config", "config"),
]

WEIGHT_CANDIDATES = [
    PROJECT_ROOT / "weights" / "groundingdino_swint_ogc.pth",
    PROJECT_ROOT / "weights" / "groundingdino_swint_ogc-2.pth",
    PROJECT_ROOT / "groundingdino_swint_ogc.pth",
    PROJECT_ROOT / "groundingdino_swint_ogc-2.pth",
]

DINO_CONFIG = next((p for p in CONFIG_CANDIDATES if p.is_file()), None)
DINO_WEIGHTS = next((p for p in WEIGHT_CANDIDATES if p.is_file()), None)

if DINO_CONFIG is None:
    raise FileNotFoundError(
        "Missing DINO config. Looked in:\n" + "\n".join(str(p) for p in CONFIG_CANDIDATES)
    )

if DINO_WEIGHTS is None:
    raise FileNotFoundError(
        "Missing DINO weights. Looked in:\n" + "\n".join(str(p) for p in WEIGHT_CANDIDATES)
    )


# =========================================================
# 3) MODEL LOADING (cached)
# =========================================================
_dino_model = None

def _get_dino_model():
    global _dino_model
    if _dino_model is not None:
        return _dino_model

    print("🔍 Loading GroundingDINO on CPU...")
    print("   config:", DINO_CONFIG)
    print("   weights:", DINO_WEIGHTS)

    _dino_model = load_model(
        model_config_path=str(DINO_CONFIG),
        model_checkpoint_path=str(DINO_WEIGHTS),
        device="cpu",
    )
    return _dino_model


def _normalize_prompt(text_prompt: Union[str, List[str]]) -> str:
    if isinstance(text_prompt, (list, tuple)):
        parts = [str(x).strip().lower() for x in text_prompt if str(x).strip()]
        caption = " . ".join(parts)
    else:
        caption = str(text_prompt).strip().lower()

    if not caption.endswith("."):
        caption += "."
    return caption


# =========================================================
# 4) MAIN API USED BY app.py
# =========================================================
def detect_and_segment(
    image_path: str,
    text_prompt: Union[str, List[str]],
    box_threshold: float = 0.15,
    text_threshold: float = 0.12,
) -> Tuple[np.ndarray, List[str]]:

    model = _get_dino_model()

    image_source, image_tensor = load_image(image_path)
    caption = _normalize_prompt(text_prompt)

    # ✅ IMPORTANT: force CPU here too
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cpu",   # <<< THIS FIXES YOUR CURRENT CRASH
    )

    if boxes is None or len(boxes) == 0:
        return image_source, []

    annotated = annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=phrases,
    )

    annotated_np = np.array(annotated, copy=False)
    if annotated_np.dtype != np.uint8:
        annotated_np = annotated_np.astype("uint8")

    labels = [str(p) for p in phrases]
    return annotated_np, labels


# ---------------------------------------------------------
# Backwards-compatible alias (prevents old app.py crashes)
# ---------------------------------------------------------
def run_detection_and_segmentation(
    image_path: str,
    text_prompt,
    box_threshold: float = 0.15,
    text_threshold: float = 0.12,
    use_sam: bool = False,
):
    return detect_and_segment(
        image_path=image_path,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
