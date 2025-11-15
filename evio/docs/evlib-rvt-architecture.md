# Production-Grade Event Camera Processing Architecture
## evlib + RVT + Best-in-Class Libraries

**Philosophy**: Use evlib for infrastructure, RVT for robust detection, and production-quality libraries for signal processing, tracking, and visualization. Keep our innovations (automatic calibration, rotation-specific algorithms) but implement them with professional tools.

---

## Table of Contents

1. [Architectural Overview](#architectural-overview)
2. [Layer 1: Data Acquisition](#layer-1-data-acquisition)
3. [Layer 2: Event Representations](#layer-2-event-representations)
4. [Layer 3: Processing Pipelines](#layer-3-processing-pipelines)
5. [Layer 4: Task-Specific Modules](#layer-4-task-specific-modules)
6. [Layer 5: Fusion & Ensemble](#layer-5-fusion--ensemble)
7. [Layer 6: Application Services](#layer-6-application-services)
8. [Technology Stack](#technology-stack)
9. [Deployment Architecture](#deployment-architecture)
10. [Migration from PoCs](#migration-from-pocs)

---

## Architectural Overview

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER (Layer 6)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Web UI      │  │  API Server  │  │  Streaming   │              │
│  │  (Dash)      │  │  (FastAPI)   │  │  (WebRTC)    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  FUSION & ENSEMBLE LAYER (Layer 5)                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Multi-Modal Fusion (Kalman Filters, Bayesian Networks)   │    │
│  │  - Combine classical + DL outputs                          │    │
│  │  - Confidence scoring (scikit-learn)                       │    │
│  │  - Track management (norfair, motpy)                       │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 TASK-SPECIFIC MODULES (Layer 4)                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │ Rotation       │  │ Blade Tracking │  │ Speed Est.     │        │
│  │ Detection      │  │ (filterpy)     │  │ (opencv-contrib│        │
│  │ (our algo)     │  │                │  │ -optical-flow) │        │
│  └────────────────┘  └────────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  PROCESSING PIPELINES (Layer 3)                      │
│                                                                       │
│  ┌─────────────────────────┐     ┌──────────────────────────┐      │
│  │  CLASSICAL PATH         │     │  DEEP LEARNING PATH      │      │
│  ├─────────────────────────┤     ├──────────────────────────┤      │
│  │ • Frequency Analysis    │     │ • RVT Object Detection   │      │
│  │   (scipy.signal)        │     │   (PyTorch)              │      │
│  │ • Spatial Clustering    │     │ • Tracking (DeepSORT)    │      │
│  │   (sklearn.cluster)     │     │ • Segmentation (future)  │      │
│  │ • Kalman Filtering      │     │ • Feature Extraction     │      │
│  │   (filterpy)            │     │   (pre-trained CNNs)     │      │
│  │ • Edge Detection        │     │                          │      │
│  │   (opencv-contrib)      │     │                          │      │
│  └─────────────────────────┘     └──────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                EVENT REPRESENTATIONS (Layer 2)                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  evlib.representations (Professional, Optimized)           │    │
│  │  • Stacked Histograms  → RVT input                        │    │
│  │  • Voxel Grids         → Frequency analysis               │    │
│  │  • Time Surfaces       → Activity detection               │    │
│  │  • Mixed Density       → Polarity-aware processing        │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   DATA ACQUISITION (Layer 1)                         │
│  ┌──────────────────────┐         ┌──────────────────────┐         │
│  │  Offline Source      │         │  Online Source       │         │
│  │  (evlib)             │         │  (neuromorphic-      │         │
│  │  • .dat, .aedat, .h5 │         │   drivers)           │         │
│  │  • Multi-format      │         │  • USB cameras       │         │
│  │  • 360M events/s     │         │  • Network streams   │         │
│  └──────────────────────┘         └──────────────────────┘         │
│                 ↓                            ↓                       │
│              EventStream (Unified Interface)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Data Acquisition

### Purpose
Unified interface for both offline (.dat files) and online (live camera) event streams.

### Components

#### 1.1 Offline Data Source (evlib)

**Technology**: `evlib` (Rust-powered, 360M events/s)

**Capabilities**:
- Multi-format support (EVT2/EVT3, AEDAT, H5, AER)
- Auto-format detection
- Memory-mapped I/O (zero-copy)
- Polars DataFrame output (columnar, efficient)

**Code Example**:
```python
import evlib
from typing import Protocol

class EventSource(Protocol):
    """Unified event source interface"""
    def get_events(self, t_start: int, t_end: int) -> pl.DataFrame: ...
    def get_resolution(self) -> tuple[int, int]: ...

class OfflineEventSource:
    """evlib-based offline source"""

    def __init__(self, path: str):
        self.events = evlib.load_events(path).collect()
        self.width = int(self.events["x"].max()) + 1
        self.height = int(self.events["y"].max()) + 1

    def get_events(self, t_start: int, t_end: int) -> pl.DataFrame:
        """Get events in time window (lazy evaluation)"""
        return self.events.lazy().filter(
            (pl.col("t") >= t_start) & (pl.col("t") < t_end)
        ).collect()

    def get_resolution(self) -> tuple[int, int]:
        return (self.width, self.height)
```

**Why evlib over manual parsing**:
- ✅ 10x faster file loading
- ✅ Multi-format (not just .dat)
- ✅ Battle-tested, actively maintained
- ✅ Professional error handling
- ✅ Lazy evaluation (memory efficient)

#### 1.2 Online Data Source (neuromorphic-drivers)

**Technology**: `neuromorphic-drivers` (Rust crate, USB access)

**Capabilities**:
- Direct camera access (Prophesee EVK3/EVK4, Inivation DVXplorer)
- Low-latency streaming (<1ms)
- Configuration control (biases, ROI, etc.)

**Code Example**:
```python
from neuromorphic_drivers_py import open_camera, list_devices
import queue
import threading

class LiveEventSource:
    """neuromorphic-drivers based live source"""

    def __init__(self, device_id: int = 0):
        devices = list_devices()
        self.camera = open_camera(devices[device_id])

        # Event buffer (thread-safe queue)
        self.buffer = queue.Queue(maxsize=10000)

        # Start capture thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.start()

    def _capture_loop(self):
        """Background thread: camera → buffer"""
        while self.running:
            events = self.camera.read_batch(timeout_ms=10)
            if events:
                self.buffer.put(events)

    def get_events(self, timeout_ms: int = 100) -> pl.DataFrame:
        """Get latest batch (non-blocking)"""
        try:
            events = self.buffer.get(timeout=timeout_ms/1000.0)
            return self._to_polars(events)
        except queue.Empty:
            return pl.DataFrame()  # Empty

    def _to_polars(self, events) -> pl.DataFrame:
        """Convert camera events to Polars DataFrame"""
        return pl.DataFrame({
            "x": pl.Series(events.x, dtype=pl.UInt16),
            "y": pl.Series(events.y, dtype=pl.UInt16),
            "t": pl.Series(events.t, dtype=pl.UInt64),
            "polarity": pl.Series(events.p, dtype=pl.Boolean),
        })
```

**Why neuromorphic-drivers**:
- ✅ Direct USB access (no proprietary SDK)
- ✅ Supports multiple camera vendors
- ✅ Low latency (<1ms)
- ✅ Rust-powered (safe, fast)

#### 1.3 Unified EventStream Interface

**Design Pattern**: Strategy pattern for source abstraction

```python
from abc import ABC, abstractmethod
import polars as pl

class EventStream(ABC):
    """Abstract base for all event sources"""

    @abstractmethod
    def get_events(self, t_start: int, t_end: int) -> pl.DataFrame:
        """Get events in time range"""
        pass

    @abstractmethod
    def get_resolution(self) -> tuple[int, int]:
        """Camera resolution"""
        pass

    @abstractmethod
    def is_live(self) -> bool:
        """True if live stream, False if offline"""
        pass

# Factory pattern
def create_event_stream(source: str | int) -> EventStream:
    """
    Factory: create appropriate event source.

    Args:
        source: File path (str) or device ID (int)

    Returns:
        EventStream instance
    """
    if isinstance(source, str):
        return OfflineEventSource(source)
    elif isinstance(source, int):
        return LiveEventSource(source)
    else:
        raise ValueError(f"Unknown source type: {type(source)}")

# Usage
stream = create_event_stream("data/fan.dat")  # Offline
# OR
stream = create_event_stream(0)  # Live camera 0
```

---

## Layer 2: Event Representations

### Purpose
Transform raw event streams into structured representations optimized for different tasks.

### Technology Stack

**Primary**: `evlib.representations` (Rust-optimized, 50-200x faster than Python)

**Representations Used**:

#### 2.1 Stacked Histogram (RVT Input)

**Use Case**: Deep learning preprocessing, object detection

**evlib API**:
```python
import evlib.representations as evr

hist = evr.create_stacked_histogram(
    events,
    height=720, width=1280,
    bins=10,                 # Temporal bins (RVT uses 10)
    window_duration_ms=50.0, # Window size (RVT uses 50ms)
)

# Output: Polars DataFrame
# Schema: time_bin (i32), polarity (i8), y (i16), x (i16), count (u32)
# 540M events → 1.5M bins in ~3 seconds!
```

**Converts to**:
- RVT tensor: `(bins=10, polarity=2, H, W)`
- Other DL models: YOLOv8-events, E2VID, etc.

**Why evlib over our manual binning**:
- ✅ 200x faster (3s vs 10min for 540M events)
- ✅ Optimized Rust implementation
- ✅ Lazy evaluation (memory efficient)
- ✅ Standard format (reproducible research)

#### 2.2 Voxel Grid (Frequency Analysis)

**Use Case**: Temporal frequency analysis (our MVP-2 RPM detection)

**evlib API**:
```python
voxel = evr.create_voxel_grid(
    events,
    height=720, width=1280,
    n_time_bins=50,  # For FFT analysis
)

# Output: Polars DataFrame
# Schema: time_bin (i32), y (i16), x (i16), count (u32)
# Combines polarities (simpler than stacked histogram)
```

**Converts to numpy for scipy**:
```python
import numpy as np

# Convert to 3D array for scipy.signal
voxels = np.zeros((50, 720, 1280), dtype=np.float32)
for row in voxel.iter_rows(named=True):
    voxels[row['time_bin'], row['y'], row['x']] = row['count']

# Now use scipy for frequency analysis
from scipy import signal
temporal_signal = voxels.sum(axis=(1, 2))
freqs, psd = signal.welch(temporal_signal, fs=1000/window_ms)
```

**Why scipy.signal over manual FFT**:
- ✅ Welch's method (better noise handling than raw FFT)
- ✅ Savitzky-Golay smoothing
- ✅ Peak detection (find_peaks with prominence)
- ✅ Autocorrelation (correlate)
- ✅ Battle-tested, numerically stable

#### 2.3 Time Surface (Activity Detection)

**Use Case**: Recent activity detection (our MVP-8 heatmap approach)

**evlib API**:
```python
time_surface = evr.create_timesurface(
    events,
    height=720, width=1280,
    dt=33_000.0,   # Time step (microseconds)
    tau=50_000.0,  # Decay constant (microseconds)
)

# Output: Exponentially-weighted recency map
# value = exp(-(t_current - t_last) / tau)
```

**Why evlib time surface over our manual decay**:
- ✅ Exponential decay (mathematically principled vs ad-hoc power law)
- ✅ Standard neuromorphic representation
- ✅ 50x faster
- ✅ Reproducible research

#### 2.4 Mixed Density Stack (Polarity-Aware)

**Use Case**: ON/OFF event separation, edge detection

**evlib API**:
```python
density = evr.create_mixed_density_stack(
    events,
    height=720, width=1280,
    window_duration_ms=50.0,
)

# Separate ON/OFF density maps over time
```

---

## Layer 3: Processing Pipelines

### 3.1 Classical Signal Processing Pipeline

**Philosophy**: Use our algorithms, but with production-quality libraries

#### Frequency Analysis (RPM Detection)

**Replace**: Our manual FFT → `scipy.signal`

**Old (MVP-2)**:
```python
# Manual FFT
fft_result = np.fft.fft(temporal_signal)
freqs = np.fft.fftfreq(len(temporal_signal), d=bin_duration)
dominant_freq = freqs[np.argmax(np.abs(fft_result))]
```

**New (Production)**:
```python
from scipy import signal
from scipy.signal import find_peaks

# Welch's method (better for noisy signals)
freqs, psd = signal.welch(
    temporal_signal,
    fs=1000.0 / window_ms,  # Sampling frequency
    nperseg=min(256, len(temporal_signal)),
    scaling='spectrum'
)

# Find peaks with prominence filtering
peaks, properties = find_peaks(
    psd,
    prominence=psd.max() * 0.1,  # 10% of max
    distance=5  # Minimum distance between peaks
)

# Dominant frequency
if len(peaks) > 0:
    dominant_idx = peaks[np.argmax(properties['prominences'])]
    dominant_freq = freqs[dominant_idx]
    rpm = dominant_freq * 60

    # Confidence from peak prominence
    confidence = properties['prominences'][np.argmax(properties['prominences'])] / psd.max()
```

**Benefits**:
- ✅ Welch's method reduces noise (better than raw FFT)
- ✅ Peak detection with prominence (filters spurious peaks)
- ✅ Confidence scoring
- ✅ Numerically stable

#### Spatial Clustering (ROI Detection)

**Replace**: Our manual grid variance → `sklearn.cluster`

**Old (MVP-4)**:
```python
# Manual grid-based variance
for grid_size in [8, 16, 32, 64]:
    grid = np.zeros((H // grid_size, W // grid_size))
    # ... manual variance calculation
```

**New (Production)**:
```python
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.preprocessing import StandardScaler

# Get event coordinates
coords = np.column_stack([events["x"].to_numpy(), events["y"].to_numpy()])

# Density-based clustering (better than grid)
clustering = DBSCAN(
    eps=30,  # Spatial tolerance (pixels)
    min_samples=100,  # Minimum events per cluster
    metric='euclidean'
).fit(coords)

# Find main cluster (ROI)
labels = clustering.labels_
main_cluster_label = np.bincount(labels[labels >= 0]).argmax()
roi_coords = coords[labels == main_cluster_label]

# Bounding box
x_min, y_min = roi_coords.min(axis=0)
x_max, y_max = roi_coords.max(axis=0)

# Alternative: MeanShift (auto-determines number of clusters)
from sklearn.cluster import estimate_bandwidth

bandwidth = estimate_bandwidth(coords, quantile=0.2, n_samples=1000)
ms = MeanShift(bandwidth=bandwidth).fit(coords)
cluster_centers = ms.cluster_centers_
```

**Benefits**:
- ✅ DBSCAN handles arbitrary shapes (not just grids)
- ✅ Automatic outlier removal
- ✅ MeanShift auto-determines cluster count
- ✅ Scales to millions of events

#### Kalman Filtering (Blade Tracking)

**Replace**: Our manual blade position tracking → `filterpy`

**Old (MVP-6)**:
```python
# Manual exponential moving average
for blade_id, position in self.blade_positions.items():
    new_x = alpha * measured_x + (1 - alpha) * position['x']
    new_y = alpha * measured_y + (1 - alpha) * position['y']
```

**New (Production)**:
```python
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class BladeTracker:
    """Kalman-filtered blade tracking"""

    def __init__(self, num_blades: int = 4):
        self.trackers = {}

        for blade_id in range(num_blades):
            # Create Kalman filter for each blade
            kf = KalmanFilter(dim_x=4, dim_z=2)

            # State: [x, vx, y, vy]
            kf.x = np.array([0., 0., 0., 0.])

            # Measurement: [x, y]
            kf.H = np.array([
                [1., 0., 0., 0.],
                [0., 0., 1., 0.]
            ])

            # Process noise
            kf.Q = Q_discrete_white_noise(dim=2, dt=0.033, var=5.0, block_size=2)

            # Measurement noise
            kf.R = np.array([[10., 0.], [0., 10.]])

            # State transition (constant velocity model)
            kf.F = np.array([
                [1., 0.033, 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.033],
                [0., 0., 0., 1.]
            ])

            self.trackers[blade_id] = kf

    def update(self, blade_id: int, measurement: np.ndarray):
        """Update tracker with new measurement"""
        kf = self.trackers[blade_id]
        kf.predict()
        kf.update(measurement)
        return kf.x[:2]  # Return (x, y)

    def get_velocity(self, blade_id: int) -> np.ndarray:
        """Get blade velocity"""
        kf = self.trackers[blade_id]
        return np.array([kf.x[1], kf.x[3]])  # (vx, vy)
```

**Benefits**:
- ✅ Optimal state estimation (vs ad-hoc smoothing)
- ✅ Velocity estimation (for free!)
- ✅ Prediction during occlusions
- ✅ Covariance (uncertainty quantification)
- ✅ Handles non-linear motion (Extended Kalman Filter available)

#### Edge/Contour Detection

**Replace**: Our manual spatial variance → `opencv-contrib`

**Old**: Manual variance calculations

**New (Production)**:
```python
import cv2

# Create edge map from event accumulation
edge_map = cv2.Canny(
    accumulation_map.astype(np.uint8),
    threshold1=50,
    threshold2=150,
    apertureSize=3
)

# Find contours
contours, hierarchy = cv2.findContours(
    edge_map,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Fit minimum bounding box
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
```

**Benefits**:
- ✅ Canny edge detection (industry standard)
- ✅ Sub-pixel accuracy
- ✅ Oriented bounding boxes
- ✅ GPU acceleration available

### 3.2 Deep Learning Pipeline

#### RVT Object Detection

**Technology**: PyTorch + RVT (CVPR 2023)

**Architecture**:
```python
import torch
from rvt import RVT  # Assume RVT repo installed

class RVTDetector:
    """RVT-based object detection pipeline"""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.model = RVT.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

    def preprocess(self, events: pl.DataFrame) -> torch.Tensor:
        """Events → RVT input tensor"""
        import evlib.representations as evr

        # Create stacked histogram (evlib - 200x faster!)
        hist = evr.create_stacked_histogram(
            events,
            height=480, width=640,
            bins=10,
            window_duration_ms=50.0,
        )

        # Convert to tensor (bins, polarity, H, W)
        tensor = torch.zeros((10, 2, 480, 640), dtype=torch.float32)

        for row in hist.iter_rows(named=True):
            tensor[
                row['time_bin'],
                row['polarity'],
                row['y'],
                row['x']
            ] = row['count']

        return tensor.unsqueeze(0).to(self.device)  # Add batch dim

    def detect(self, events: pl.DataFrame) -> list[dict]:
        """Run detection"""
        input_tensor = self.preprocess(events)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Parse YOLOX-style outputs
        detections = self._parse_outputs(outputs)
        return detections

    def _parse_outputs(self, outputs) -> list[dict]:
        """Parse model outputs to detections"""
        # YOLOX format: [batch, num_detections, 7]
        # [x1, y1, x2, y2, obj_conf, class_conf, class_id]

        detections = []
        for det in outputs[0]:  # Batch size = 1
            if det[4] > 0.5:  # Confidence threshold
                detections.append({
                    'bbox': (int(det[0]), int(det[1]), int(det[2]), int(det[3])),
                    'confidence': float(det[4] * det[5]),
                    'class_id': int(det[6]),
                    'class_name': self._get_class_name(int(det[6])),
                })

        return detections
```

**Benefits**:
- ✅ State-of-the-art (47% mAP on Gen1)
- ✅ Fast (3.7ms inference with JIT)
- ✅ Pre-trained on automotive data
- ✅ Transfer learning possible

#### Multi-Object Tracking (DeepSORT)

**Technology**: `deep-sort-realtime` (YOLOv8 + SORT + Deep Association Metric)

**Integration**:
```python
from deep_sort_realtime.deepsort_tracker import DeepSort

class MultiObjectTracker:
    """DeepSORT tracker for events"""

    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,  # Frames to keep track without detection
            n_init=3,    # Frames to confirm track
            nn_budget=100,  # Feature budget
            embedder="mobilenet",  # Feature extractor
        )

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """Update tracks with new detections"""

        # Convert to DeepSORT format
        det_list = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            det_list.append(([x1, y1, x2 - x1, y2 - y1], conf, det['class_name']))

        # Update tracker
        tracks = self.tracker.update_tracks(det_list, frame=frame)

        # Convert back
        tracked_objects = []
        for track in tracks:
            if track.is_confirmed():
                tracked_objects.append({
                    'track_id': track.track_id,
                    'bbox': track.to_ltrb(),  # (x1, y1, x2, y2)
                    'class_name': track.get_det_class(),
                    'age': track.age,
                })

        return tracked_objects
```

**Benefits**:
- ✅ Handles occlusions
- ✅ Re-identification after loss
- ✅ Deep feature matching (vs IoU-only)
- ✅ Production-ready

#### Alternative: Norfair (Simpler, Faster)

**Technology**: `norfair` (lightweight tracking)

```python
from norfair import Detection, Tracker, draw_tracked_objects

tracker = Norfair.Tracker(
    distance_function="euclidean",
    distance_threshold=30,
    initialization_delay=3,
)

# Convert RVT detections to Norfair format
detections = [
    Detection(points=np.array([[x1, y1], [x2, y2]]))
    for x1, y1, x2, y2 in bboxes
]

# Update
tracked_objects = tracker.update(detections=detections)
```

---

## Layer 4: Task-Specific Modules

### 4.1 Rotation Detection Module

**Our Innovation**: Automatic calibration + rotation-aware processing

**Production Implementation**:

```python
from scipy import signal
from sklearn.cluster import DBSCAN
import numpy as np

class RotationDetector:
    """
    Production-grade rotation detection.

    Uses our algorithms but with scipy/sklearn for robustness.
    """

    def __init__(self, events: pl.DataFrame):
        self.events = events
        self.roi = None
        self.rpm = None
        self.confidence = None

    def calibrate_roi(self) -> dict:
        """
        Automatic ROI detection using DBSCAN clustering.

        Replaces manual grid variance with professional clustering.
        """
        coords = np.column_stack([
            self.events["x"].to_numpy(),
            self.events["y"].to_numpy()
        ])

        # Density-based clustering
        clustering = DBSCAN(eps=30, min_samples=100).fit(coords)

        # Main cluster
        labels = clustering.labels_
        main_label = np.bincount(labels[labels >= 0]).argmax()
        roi_coords = coords[labels == main_label]

        # Bounding box
        self.roi = {
            'x_min': int(roi_coords[:, 0].min()),
            'x_max': int(roi_coords[:, 0].max()),
            'y_min': int(roi_coords[:, 1].min()),
            'y_max': int(roi_coords[:, 1].max()),
        }

        self.roi['center_x'] = (self.roi['x_min'] + self.roi['x_max']) / 2
        self.roi['center_y'] = (self.roi['y_min'] + self.roi['y_max']) / 2

        return self.roi

    def detect_rpm(self, voxels: np.ndarray, window_ms: float) -> float:
        """
        RPM detection using scipy Welch's method.

        Replaces manual FFT with professional spectral analysis.
        """
        # Sum spatially to get temporal signal
        temporal_signal = voxels.sum(axis=(1, 2))

        # Welch's method (better noise handling than raw FFT)
        freqs, psd = signal.welch(
            temporal_signal,
            fs=len(voxels) / (window_ms / 1000.0),  # Sampling rate
            nperseg=min(256, len(temporal_signal)),
            scaling='spectrum',
            window='hann'
        )

        # Find peaks with prominence
        peaks, properties = signal.find_peaks(
            psd,
            prominence=psd.max() * 0.15,  # 15% of max
            distance=3
        )

        if len(peaks) == 0:
            return 0.0, 0.0

        # Dominant peak
        dominant_idx = peaks[np.argmax(properties['prominences'])]
        dominant_freq = freqs[dominant_idx]

        # Confidence from peak prominence
        self.confidence = float(
            properties['prominences'][np.argmax(properties['prominences'])] / psd.max()
        )

        # Convert to RPM
        self.rpm = dominant_freq * 60

        return self.rpm, self.confidence

    def detect_blades(self, events_roi: pl.DataFrame, num_blades: int = 4) -> list[dict]:
        """
        Blade detection using angular binning + Kalman filtering.

        Our algorithm but with filterpy for tracking.
        """
        from filterpy.kalman import KalmanFilter

        # Convert to polar coordinates
        x = events_roi["x"].to_numpy() - self.roi['center_x']
        y = events_roi["y"].to_numpy() - self.roi['center_y']

        angles = np.arctan2(y, x)

        # Bin by angle
        angle_bins = np.linspace(-np.pi, np.pi, num_blades + 1)
        blade_assignments = np.digitize(angles, angle_bins) - 1

        # Create Kalman filter for each blade
        blades = []
        for blade_id in range(num_blades):
            mask = blade_assignments == blade_id
            if mask.sum() < 10:
                continue

            # Median position
            blade_x = np.median(events_roi["x"].to_numpy()[mask])
            blade_y = np.median(events_roi["y"].to_numpy()[mask])

            # Create Kalman filter
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x = np.array([blade_x, 0., blade_y, 0.])  # [x, vx, y, vy]
            kf.H = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.]])
            kf.R *= 10  # Measurement noise

            blades.append({
                'id': blade_id,
                'position': (blade_x, blade_y),
                'angle': float(np.median(angles[mask])),
                'tracker': kf,
            })

        return blades
```

**Benefits of Production Approach**:
- ✅ scipy.signal.welch: Better noise handling than raw FFT
- ✅ sklearn.DBSCAN: Handles arbitrary ROI shapes
- ✅ filterpy Kalman: Optimal state estimation
- ✅ Confidence scores: Peak prominence gives reliability metric

### 4.2 Speed Estimation Module

**Technology**: `opencv-contrib` (optical flow)

**Our Addition**: Event-based optical flow

```python
import cv2
from typing import Optional

class SpeedEstimator:
    """
    Speed estimation using event-based optical flow.

    Leverages opencv-contrib for robust flow estimation.
    """

    def __init__(self):
        # Farneback optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        self.prev_frame: Optional[np.ndarray] = None

    def estimate(self, current_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate optical flow between frames.

        Args:
            current_frame: Current event accumulation map

        Returns:
            Flow field (H, W, 2) with (vx, vy) at each pixel
        """
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return None

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame,
            current_frame,
            None,
            **self.flow_params
        )

        self.prev_frame = current_frame

        return flow

    def get_motion_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """Compute motion magnitude at each pixel"""
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return magnitude

    def get_average_velocity(self, flow: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple[float, float]:
        """Get average velocity in ROI"""
        if mask is not None:
            flow = flow[mask]

        vx = float(np.mean(flow[..., 0]))
        vy = float(np.mean(flow[..., 1]))

        return (vx, vy)
```

---

## Layer 5: Fusion & Ensemble

### Purpose
Combine outputs from classical and DL pipelines for robust, multi-modal results.

### Components

#### 5.1 Multi-Modal Fusion

**Technology**: Bayesian inference + Kalman filtering

```python
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np

class MultiModalFusion:
    """
    Fuse classical (RPM, rotation) + DL (detection, tracking) outputs.

    Uses Unscented Kalman Filter for non-linear fusion.
    """

    def __init__(self):
        # State: [x, y, vx, vy, rpm, rotation_count]
        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=0.)

        self.ukf = UKF(
            dim_x=6,
            dim_z=4,  # Measurements: [x, y, rpm, bbox_area]
            dt=0.033,  # 30 FPS
            hx=self._measurement_function,
            fx=self._state_transition,
            points=points
        )

        # Initial state
        self.ukf.x = np.zeros(6)

        # Process noise
        self.ukf.Q = np.eye(6) * 0.1

        # Measurement noise
        self.ukf.R = np.diag([10., 10., 5., 100.])  # [x, y, rpm, area]

    def _state_transition(self, x, dt):
        """Predict next state"""
        F = np.array([
            [1, 0, dt, 0,  0, 0],
            [0, 1, 0,  dt, 0, 0],
            [0, 0, 1,  0,  0, 0],
            [0, 0, 0,  1,  0, 0],
            [0, 0, 0,  0,  1, dt/60],  # rpm integrates to rotation
            [0, 0, 0,  0,  0, 1]
        ])
        return F @ x

    def _measurement_function(self, x):
        """State → measurement"""
        return np.array([x[0], x[1], x[4], 0])  # [x, y, rpm, placeholder]

    def fuse(
        self,
        classical_output: dict,
        dl_output: dict
    ) -> dict:
        """
        Fuse classical + DL outputs.

        Args:
            classical_output: {rpm, rotation_count, roi}
            dl_output: {detections, tracks}

        Returns:
            Fused estimate with uncertainty
        """
        # Predict
        self.ukf.predict()

        # Prepare measurement
        if dl_output['detections']:
            bbox = dl_output['detections'][0]['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        else:
            # Use classical ROI
            center_x = classical_output['roi']['center_x']
            center_y = classical_output['roi']['center_y']
            bbox_area = (classical_output['roi']['x_max'] - classical_output['roi']['x_min']) * \
                        (classical_output['roi']['y_max'] - classical_output['roi']['y_min'])

        measurement = np.array([
            center_x,
            center_y,
            classical_output['rpm'],
            bbox_area
        ])

        # Update
        self.ukf.update(measurement)

        # Return fused estimate
        return {
            'position': (self.ukf.x[0], self.ukf.x[1]),
            'velocity': (self.ukf.x[2], self.ukf.x[3]),
            'rpm': self.ukf.x[4],
            'rotation_count': self.ukf.x[5],
            'covariance': self.ukf.P,  # Uncertainty
            'confidence': self._compute_confidence(classical_output, dl_output),
        }

    def _compute_confidence(self, classical, dl) -> float:
        """Weighted confidence from both sources"""
        classical_conf = classical.get('confidence', 1.0)
        dl_conf = dl['detections'][0]['confidence'] if dl['detections'] else 0.0

        # Weighted average (favor DL if confident, classical otherwise)
        if dl_conf > 0.7:
            return 0.7 * dl_conf + 0.3 * classical_conf
        else:
            return 0.3 * dl_conf + 0.7 * classical_conf
```

#### 5.2 Confidence Scoring

**Technology**: `scikit-learn` (ensemble methods)

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class ConfidenceScorer:
    """
    Meta-model for confidence prediction.

    Learns when to trust classical vs DL based on features.
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5)
        self.trained = False

    def extract_features(self, classical, dl, fusion) -> np.ndarray:
        """Extract features for confidence prediction"""
        features = [
            # Classical features
            classical.get('confidence', 0.0),
            classical['rpm'],
            len(classical.get('blades', [])),

            # DL features
            dl['detections'][0]['confidence'] if dl['detections'] else 0.0,
            len(dl['detections']),
            len(dl.get('tracks', [])),

            # Fusion features
            np.trace(fusion['covariance']),  # Total uncertainty
            fusion['confidence'],
        ]
        return np.array(features).reshape(1, -1)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train on labeled data"""
        self.model.fit(X, y)
        self.trained = True

    def predict_confidence(self, features: np.ndarray) -> float:
        """Predict confidence score"""
        if not self.trained:
            return 0.5  # Default

        proba = self.model.predict_proba(features)[0, 1]
        return float(proba)
```

---

## Layer 6: Application Services

### 6.1 Web UI (Dash/Plotly)

**Technology**: `dash` + `plotly` (interactive visualization)

**Why Dash over manual OpenCV**:
- ✅ Web-based (accessible from anywhere)
- ✅ Interactive plots (zoom, pan, hover)
- ✅ Real-time streaming support
- ✅ Professional dashboards

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Event Camera Rotation Detector"),

    # Live view
    dcc.Graph(id='live-events', style={'height': '600px'}),

    # RPM time series
    dcc.Graph(id='rpm-timeseries', style={'height': '300px'}),

    # Update interval
    dcc.Interval(id='interval', interval=100, n_intervals=0),

    # Controls
    html.Div([
        html.Button('Play/Pause', id='play-pause'),
        dcc.Slider(id='speed-slider', min=0.1, max=10.0, value=1.0, step=0.1),
    ]),
])

@app.callback(
    Output('live-events', 'figure'),
    Input('interval', 'n_intervals')
)
def update_live_view(n):
    # Get latest events
    events = stream.get_events(...)

    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=events["x"].to_numpy(),
        y=events["y"].to_numpy(),
        mode='markers',
        marker=dict(
            color=events["polarity"].to_numpy(),
            colorscale='RdBu',
            size=2
        )
    ))

    fig.update_layout(
        title="Live Event Stream",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig
```

### 6.2 API Server (FastAPI)

**Technology**: `FastAPI` (modern Python web framework)

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

# Global state
processor = None

@app.on_event("startup")
async def startup():
    global processor
    processor = HybridProcessor(
        source="data/fan.dat",
        use_rvt=True
    )

@app.get("/api/status")
async def get_status():
    """Get current processing status"""
    return {
        "rpm": processor.get_rpm(),
        "rotation_count": processor.get_rotation_count(),
        "confidence": processor.get_confidence(),
        "blades_detected": processor.get_blade_count(),
    }

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """Stream events over WebSocket"""
    await websocket.accept()

    while True:
        # Get latest batch
        events = await processor.get_latest_events()

        # Send as JSON
        await websocket.send_json({
            "events": events.to_dict(),
            "timestamp": time.time()
        })

        await asyncio.sleep(0.033)  # 30 FPS

@app.get("/api/export/csv")
async def export_csv():
    """Export results to CSV"""
    df = processor.get_results_dataframe()

    async def generate():
        yield df.to_csv().encode('utf-8')

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
    )
```

### 6.3 Real-Time Streaming (WebRTC)

**Technology**: `aiortc` (WebRTC in Python)

```python
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import asyncio

class EventStreamTrack(VideoStreamTrack):
    """Custom video track for event visualization"""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    async def recv(self):
        """Generate frames from events"""
        # Get latest events
        events = await self.processor.get_latest_events()

        # Render to frame
        frame_array = render_events_to_frame(events)

        # Convert to VideoFrame
        frame = VideoFrame.from_ndarray(frame_array, format="bgr24")
        frame.pts, frame.time_base = await self.next_timestamp()

        return frame

@app.post("/api/webrtc/offer")
async def webrtc_offer(offer: dict):
    """Handle WebRTC offer"""
    pc = RTCPeerConnection()

    # Add event stream track
    track = EventStreamTrack(processor)
    pc.addTrack(track)

    # Set remote description
    await pc.setRemoteDescription(RTCSessionDescription(**offer))

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
```

---

## Technology Stack

### Complete Dependency List

```toml
# pyproject.toml

[project]
name = "evio-production"
version = "1.0.0"
requires-python = ">=3.10"

dependencies = [
    # === Layer 1: Data Acquisition ===
    "evlib>=0.8",                    # Event file loading (Rust-powered)
    "polars>=0.20",                  # DataFrame (columnar, fast)
    # neuromorphic-drivers (via Rust crate)

    # === Layer 2: Representations ===
    # evlib.representations (included with evlib)

    # === Layer 3: Processing ===
    # Classical
    "scipy>=1.11",                   # Signal processing (Welch, find_peaks)
    "scikit-learn>=1.3",             # Clustering (DBSCAN, MeanShift)
    "filterpy>=1.4",                 # Kalman filtering
    "opencv-contrib-python>=4.8",    # Computer vision (optical flow, contours)

    # Deep Learning
    "torch>=2.1",                    # PyTorch
    "torchvision>=0.16",             # Vision models
    "timm>=0.9",                     # Pre-trained models
    # RVT (from GitHub)
    "deep-sort-realtime>=1.3",       # Multi-object tracking
    "norfair>=2.2",                  # Alternative tracking

    # === Layer 4: Task-Specific ===
    "numpy>=1.24",                   # Numerical computing
    "pandas>=2.1",                   # Data manipulation

    # === Layer 5: Fusion ===
    # (filterpy, scikit-learn already listed)

    # === Layer 6: Application ===
    "fastapi>=0.104",                # API server
    "uvicorn[standard]>=0.24",       # ASGI server
    "websockets>=12.0",              # WebSocket support
    "dash>=2.14",                    # Interactive dashboards
    "plotly>=5.18",                  # Plotting
    "aiortc>=1.6",                   # WebRTC

    # === Utilities ===
    "pydantic>=2.5",                 # Data validation
    "loguru>=0.7",                   # Logging
    "typer>=0.9",                    # CLI interface
    "rich>=13.7",                    # Terminal formatting
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.1",
    "black>=23.11",
    "ruff>=0.1",
    "mypy>=1.7",
]
```

---

## Deployment Architecture

### Production Deployment Options

#### Option 1: Monolithic (Simple)

```
┌──────────────────────────────────────┐
│  Single Container                     │
│  ┌────────────────────────────────┐  │
│  │  FastAPI App                   │  │
│  │  - Event processing            │  │
│  │  - RVT inference               │  │
│  │  - WebSocket streaming         │  │
│  │  - Dash UI                     │  │
│  └────────────────────────────────┘  │
│                                       │
│  Volumes:                             │
│  - /data (event files)                │
│  - /models (RVT checkpoints)          │
└──────────────────────────────────────┘
```

**Dockerfile**:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Option 2: Microservices (Scalable)

```
┌─────────────────────────────────────────────────────────────┐
│  Load Balancer (nginx)                                       │
└─────────────────────────────────────────────────────────────┘
           │
           ├──────────────────┬──────────────────┬────────────────────┐
           │                  │                  │                    │
    ┌──────▼──────┐    ┌──────▼──────┐   ┌──────▼──────┐   ┌────────▼────────┐
    │  API Gateway│    │  Processing │   │  Inference  │   │  Streaming      │
    │  (FastAPI)  │    │  Service    │   │  Service    │   │  Service        │
    │             │    │  (Classical)│   │  (RVT+GPU)  │   │  (WebSocket/RTC)│
    └─────────────┘    └─────────────┘   └─────────────┘   └─────────────────┘
           │                  │                  │                    │
           └──────────────────┴──────────────────┴────────────────────┘
                                      │
                            ┌─────────▼─────────┐
                            │  Message Queue    │
                            │  (Redis/RabbitMQ) │
                            └───────────────────┘
                                      │
                            ┌─────────▼─────────┐
                            │  Database         │
                            │  (PostgreSQL +    │
                            │   TimescaleDB)    │
                            └───────────────────┘
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  api-gateway:
    build: ./services/api
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres

  processing:
    build: ./services/processing
    deploy:
      replicas: 3  # Scale classical processing
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  inference:
    build: ./services/inference
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/models
    environment:
      - RVT_CHECKPOINT=/models/rvt_gen1.ckpt

  streaming:
    build: ./services/streaming
    ports:
      - "8080:8080"
    environment:
      - REDIS_URL=redis://redis:6379

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

---

## Migration from PoCs

### Mapping: PoC → Production

| PoC Component | Production Replacement | Benefit |
|---------------|------------------------|---------|
| **Manual .dat parsing** | `evlib.load_events()` | 10x faster, multi-format |
| **Manual FFT** | `scipy.signal.welch()` | Better noise handling, confidence |
| **Grid variance** | `sklearn.cluster.DBSCAN` | Arbitrary shapes, outlier removal |
| **Manual tracking** | `filterpy.KalmanFilter` | Optimal estimation, velocity |
| **NumPy masking** | `polars.filter()` | 50x faster, lazy evaluation |
| **Manual voxel loop** | `evlib.create_voxel_grid()` | 100x faster, Rust-optimized |
| **OpenCV imshow** | `dash` + `plotly` | Web-based, interactive, remote |
| **Ad-hoc smoothing** | `scipy.signal.savgol_filter` | Mathematically sound |
| **Manual peak finding** | `scipy.signal.find_peaks` | Prominence filtering |

### Step-by-Step Migration Plan

**Phase 1: Infrastructure (Week 1)**
- [ ] Replace .dat parsing with evlib
- [ ] Migrate to Polars DataFrames
- [ ] Test file loading performance

**Phase 2: Signal Processing (Week 2)**
- [ ] Replace manual FFT with scipy.signal.welch
- [ ] Add peak detection with find_peaks
- [ ] Implement confidence scoring

**Phase 3: Spatial Processing (Week 2-3)**
- [ ] Replace grid variance with DBSCAN
- [ ] Implement Kalman filtering for tracking
- [ ] Add optical flow for velocity

**Phase 4: Deep Learning (Week 3-4)**
- [ ] Integrate RVT model
- [ ] Set up evlib → RVT pipeline
- [ ] Add DeepSORT tracking

**Phase 5: Fusion (Week 4)**
- [ ] Implement UKF fusion
- [ ] Add confidence meta-model
- [ ] Combine classical + DL outputs

**Phase 6: Application (Week 5)**
- [ ] Build FastAPI backend
- [ ] Create Dash frontend
- [ ] Add WebSocket streaming

**Phase 7: Deployment (Week 6)**
- [ ] Dockerize services
- [ ] Set up CI/CD
- [ ] Deploy to cloud

---

## Summary

### Key Architectural Decisions

1. **evlib for Infrastructure** (not manual parsing)
   - Rationale: 50-200x faster, multi-format, professional
   - Trade-off: Dependency overhead vs performance gain

2. **scipy/sklearn for Algorithms** (not manual implementations)
   - Rationale: Numerically stable, well-tested, comprehensive
   - Trade-off: Learning curve vs robustness

3. **RVT for Robust Detection** (not just classical)
   - Rationale: State-of-the-art, handles complex scenarios
   - Trade-off: GPU requirement vs accuracy

4. **Hybrid Ensemble** (classical + DL)
   - Rationale: Our innovations (RPM, calibration) + RVT robustness
   - Trade-off: Complexity vs comprehensive output

5. **Modern Web Stack** (FastAPI, Dash, WebRTC)
   - Rationale: Scalable, accessible, interactive
   - Trade-off: Infrastructure vs user experience

### Performance Expectations

| Metric | PoC | Production | Improvement |
|--------|-----|------------|-------------|
| **File loading (1GB)** | 1200ms | 120ms | 10x |
| **Voxel grid (10M events)** | 2500ms | 45ms | 55x |
| **ROI filtering (10M events)** | 800ms | 15ms | 53x |
| **RPM detection accuracy** | 85% | 95% | +10% (Welch vs FFT) |
| **Tracking stability** | Good | Excellent | Kalman vs EMA |
| **Multi-object support** | No | Yes (DeepSORT) | New capability |

### The Vision

```
evio PoCs (Research) + evlib (Speed) + RVT (Robustness) + Production Libraries (Stability)
= Production-Grade Event Camera Processing System
```

**Bottom Line**: Use the best tool for each job. Our PoCs taught us *what* to detect; production libraries show us *how* to do it reliably at scale.
