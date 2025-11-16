"""RVT detector plugin packaged for the UV workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import collections
import os
import polars as pl
import torch
from omegaconf import DictConfig

from rvt_adapter import (
    build_hydra_config,
    determine_device,
    ensure_repo_on_path,
    events_to_hist_tensor,
    resolve_checkpoint_path,
    resolve_repo_path,
)


@dataclass
class RVTDetection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int


class RVTDetectorPlugin:
    """DetectorPlugin-compatible wrapper around RVT."""

    name = "RVT Detector"
    key = "9"
    description = "Recurrent Vision Transformer (YOLOX head)"

    def __init__(
        self,
        rvt_repo: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        dataset_name: str = "gen1",
        experiment: str = "small",
        histogram_bins: int = 10,
        window_duration_ms: float = 50.0,
        device: Optional[str] = None,
        hydra_overrides: Optional[Sequence[str]] = None,
    ) -> None:
        self.rvt_repo = resolve_repo_path(rvt_repo)
        ensure_repo_on_path(self.rvt_repo)

        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)

        self.histogram_bins = histogram_bins
        self.window_duration_ms = window_duration_ms
        self.device = determine_device(device)

        from config.modifier import dynamically_modify_train_config  # type: ignore
        from models.detection.yolox.utils.boxes import postprocess  # type: ignore
        from modules.detection import Module as RVTModule  # type: ignore

        self._modifier = dynamically_modify_train_config
        self._postprocess = postprocess
        self._module_cls = RVTModule

        self.config = build_hydra_config(
            repo=self.rvt_repo,
            dataset_name=dataset_name,
            experiment=experiment,
            checkpoint_path=self.checkpoint_path,
            extra_overrides=hydra_overrides,
        )
        self._modifier(self.config)
        self.sensor_hw = self._sensor_hw_from_config(self.config)
        self.num_classes = int(self.config.model.head.num_classes)

        self.detector, self.input_padder = self._load_detector()
        self.detector.to(self.device)
        self.detector.eval()

        self.prev_states = None
        self.dtype = next(self.detector.parameters()).dtype

    @staticmethod
    def _sensor_hw_from_config(config: DictConfig) -> tuple[int, int]:
        hw = tuple(config.dataset.resolution_hw)
        if config.dataset.downsample_by_factor_2:
            hw = (hw[0] // 2, hw[1] // 2)
        return hw

    def _load_detector(self):
        os.environ.setdefault("TORCH_WEIGHTS_ONLY", "0")
        with torch.serialization.safe_globals(
            [
                getattr,
                torch.optim.lr_scheduler.OneCycleLR,
                torch.optim.AdamW,
                collections.defaultdict,
                dict,
            ]
        ):
            module = self._module_cls.load_from_checkpoint(
                checkpoint_path=str(self.checkpoint_path),
                full_config=self.config,
                weights_only=False,
            )
        module = module.to(self.device)
        module.eval()
        return module.mdl, module.input_padder

    def reset_states(self) -> None:
        self.prev_states = None

    def process(self, events: pl.DataFrame | pl.LazyFrame) -> List[RVTDetection]:
        tensor = self._events_to_tensor(events).to(device=self.device, dtype=self.dtype)
        tensor = self.input_padder.pad_tensor_ev_repr(tensor)
        with torch.inference_mode():
            outputs, _, states = self.detector(
                tensor, previous_states=self.prev_states
            )
        outputs = outputs.clone()
        self.prev_states = states
        processed = self._postprocess(
            prediction=outputs,
            num_classes=self.num_classes,
            conf_thre=self.config.model.postprocess.confidence_threshold,
            nms_thre=self.config.model.postprocess.nms_threshold,
        )
        return self._format_detections(processed)

    def _events_to_tensor(
        self, events: pl.DataFrame | pl.LazyFrame
    ) -> torch.Tensor:
        return events_to_hist_tensor(
            events,
            sensor_hw=self.sensor_hw,
            bins=self.histogram_bins,
            window_duration_ms=self.window_duration_ms,
        )

    @staticmethod
    def _format_detections(
        det_list: Iterable[Optional[torch.Tensor]],
    ) -> List[RVTDetection]:
        batches = list(det_list)
        if not batches or batches[0] is None or batches[0].numel() == 0:
            return []
        detections = []
        for det in batches[0].detach().cpu().tolist():
            x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det
            detections.append(
                RVTDetection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(obj_conf * cls_conf),
                    class_id=int(cls_id),
                )
            )
        return detections
