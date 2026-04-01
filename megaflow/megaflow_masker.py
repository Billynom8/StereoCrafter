"""
megaflow_masker.py - Standalone inference wrapper for MegaFlow optical flow estimation.

Runtime dependencies: torch, torchvision, safetensors, huggingface_hub, numpy

Usage:
    from megaflow.megaflow_masker import MegaFlowMasker

    masker = MegaFlowMasker.from_pretrained("megaflow-flow")
    flow = masker.get_flow(frame1, frame2)  # [B, 2, H, W]
"""

import os
import logging
from typing import Union

import torch
import torch.nn.functional as F

from .model import MegaFlow

logger = logging.getLogger(__name__)


class MegaFlowMasker:
    """Lightweight wrapper around MegaFlow for two-frame optical flow inference.

    Handles weight loading (HuggingFace or local), input format auto-detection,
    and inference with automatic mixed precision.

    Attributes:
        model: The underlying MegaFlow nn.Module in eval mode.
        device: Device string (e.g. "cuda", "cpu").
    """

    def __init__(self, model: MegaFlow, device: str):
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, device: str = "cuda") -> "MegaFlowMasker":
        """Load a MegaFlow model from HuggingFace or a local checkpoint.

        Args:
            model_name_or_path: Either a HuggingFace model name (e.g.
                ``"megaflow-flow"``) or a path to a local ``.safetensors``,
                ``.pth``, or ``.bin`` file.
            device: Target device string.

        Returns:
            A ready-to-use MegaFlowMasker instance.
        """
        model = cls._make_model()
        cls._load_weights(model, model_name_or_path, device)
        model = model.to(device).eval()
        return cls(model=model, device=device)

    @torch.inference_mode()
    def get_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Compute optical flow between two frames.

        Args:
            frame1: First frame, shape ``[B, C, H, W]`` or ``[C, H, W]``.
                Accepts ``uint8 [0, 255]`` or ``float [0, 1]`` (auto-detected).
            frame2: Second frame, same shape and format as *frame1*.

        Returns:
            Optical flow tensor of shape ``[B, 2, H, W]`` in the model's
            native resolution.
        """
        squeeze = frame1.ndim == 3
        if squeeze:
            frame1 = frame1.unsqueeze(0)
            frame2 = frame2.unsqueeze(0)

        B, C, H, W = frame1.shape

        if frame2.shape != (B, C, H, W):
            raise ValueError(f"frame1 shape {frame1.shape} and frame2 shape {frame2.shape} are incompatible")

        # Convert to float and auto-detect range
        frame1 = frame1.float()
        frame2 = frame2.float()
        if frame1.max() <= 1.0:
            frame1 = frame1 * 255.0
            frame2 = frame2 * 255.0

        imgs = torch.stack([frame1, frame2], dim=1)  # [B, 2, C, H, W]
        imgs = imgs.to(self.device)

        # Autocast dtype selection
        compute_dtype = torch.float32
        if self.device == "cuda":
            major, _ = torch.cuda.get_device_capability()
            compute_dtype = torch.bfloat16 if major >= 8 else torch.float16

        with torch.autocast(device_type=self.device, dtype=compute_dtype, enabled=(self.device == "cuda")):
            results = self.model(imgs, num_reg_refine=1)

        flow = results["flow_preds"][-1]  # [B, T-1, 2, H, W]
        flow = flow[:, 0]  # squeeze T-1 -> [B, 2, H, W]

        if squeeze:
            flow = flow.squeeze(0)  # [2, H, W]

        return flow

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _make_model() -> MegaFlow:
        return MegaFlow(fuse_cnn=True, use_temporal_attn=True, fix_width=True)

    @classmethod
    def _load_weights(cls, model: MegaFlow, source: str, device: str) -> None:
        if os.path.isfile(source):
            cls._load_local_weights(model, source)
        else:
            cls._load_hf_weights(model, source, device)

    @staticmethod
    def _load_local_weights(model: MegaFlow, path: str) -> None:
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(path, device="cpu")
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _load_hf_weights(model: MegaFlow, model_name: str, device: str) -> None:
        from huggingface_hub import hf_hub_download

        hf_models = getattr(MegaFlow, "_HF_MODELS", {})
        if model_name in hf_models:
            info = hf_models[model_name]
            ckpt_path = hf_hub_download(repo_id=info["repo_id"], filename=info["filename"])
        else:
            ckpt_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")

        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_path), device="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretrained weights from %s", ckpt_path)
