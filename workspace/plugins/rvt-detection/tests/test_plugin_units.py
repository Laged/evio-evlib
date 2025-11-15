import torch
from omegaconf import OmegaConf

from rvt_detector.plugin import RVTDetection, RVTDetectorPlugin


def test_sensor_hw_from_config_handles_downsampling():
    config = OmegaConf.create(
        {"dataset": {"resolution_hw": [720, 1280], "downsample_by_factor_2": True}}
    )
    assert RVTDetectorPlugin._sensor_hw_from_config(config) == (360, 640)


def test_sensor_hw_from_config_without_downsampling():
    config = OmegaConf.create(
        {"dataset": {"resolution_hw": [480, 640], "downsample_by_factor_2": False}}
    )
    assert RVTDetectorPlugin._sensor_hw_from_config(config) == (480, 640)


def test_format_detections_handles_empty_batches():
    assert RVTDetectorPlugin._format_detections([None]) == []
    assert RVTDetectorPlugin._format_detections([]) == []


def test_format_detections_maps_tensors_to_dataclass():
    tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.9, 0.5, 2.0]])
    detections = RVTDetectorPlugin._format_detections([tensor])
    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, RVTDetection)
    assert det.bbox == (1.0, 2.0, 3.0, 4.0)
    assert det.class_id == 2
    assert det.confidence == 0.45
