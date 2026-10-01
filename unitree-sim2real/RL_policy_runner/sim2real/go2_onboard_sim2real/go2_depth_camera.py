"""Local D435i input for the Go2 rough depth student (no training noise/blur).

Shares crop/resize/range conversion with the MuJoCo deployment path. Importing
this module does not open a camera or import the Unitree SDK.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_depth import (
    convert_image_plane_depth_to_camera_distance,
    depth_preview_uint8,
    intrinsics_from_isaaclab_pinhole_cfg,
    preprocess_depth_image,
)

TRAINING_INTRINSICS = intrinsics_from_isaaclab_pinhole_cfg(
    width=106, height=60, focal_length=11.041, horizontal_aperture=20.955, vertical_aperture=12.240
)

# Mount metadata only: these do not change image rays or compensate extrinsics.
# Positions are reported relative to Go2 base; verify the depth optical origin.
TRAINING_CAMERA_POSITION = (0.34, 0.0, 0.06)
REPORTED_CAMERA_POSITION = (0.090 + 0.32715, 0.0 - 0.00003, 0.045 + 0.04297)
TRAINING_CAMERA_EULER_INPUT_DEG = (180.0, 60.0, -90.0)
REPORTED_CAMERA_EULER_INPUT_DEG = (180.0, 90.0, -90.0)
REPORTED_CAMERA_ROTATION_VERIFIED = False


def _quat_from_euler_xyz_deg(roll_deg, pitch_deg, yaw_deg):
    """Match the task's ROS optical-offset helper, including its negated qz."""
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return (
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        -(sy * cr * cp - cy * sr * sp),
    )


def camera_mount_metadata():
    """Record physical/training mounts separately in saved camera-check samples."""
    return {
        "training_camera_position_m": np.asarray(TRAINING_CAMERA_POSITION),
        "reported_camera_position_m": np.asarray(REPORTED_CAMERA_POSITION),
        "training_camera_quaternion_wxyz": np.asarray(_quat_from_euler_xyz_deg(*TRAINING_CAMERA_EULER_INPUT_DEG)),
        "reported_camera_quaternion_wxyz": np.asarray(_quat_from_euler_xyz_deg(*REPORTED_CAMERA_EULER_INPUT_DEG)),
        "reported_camera_euler_helper_input_deg": np.asarray(REPORTED_CAMERA_EULER_INPUT_DEG),
        "reported_camera_rotation_verified": np.asarray(REPORTED_CAMERA_ROTATION_VERIFIED),
        "extrinsic_compensation_applied": np.asarray(False),
    }


def preprocess_range_image(raw_range: np.ndarray) -> np.ndarray:
    """The existing sim2sim transform, including its zero-invalid convention."""
    if raw_range.shape != (60, 106):
        raise ValueError(f"Expected a 60x106 ray-distance image, got {raw_range.shape}")
    return preprocess_depth_image(
        raw_range,
        crop_top=0,
        crop_bottom=2,
        crop_left=4,
        crop_right=4,
        resize=(58, 87),
        normalize=True,
        max_distance=2.0,
        clip_to_max_distance=True,
    )


class DepthRemapper:
    """Sample rectified native depth at the training camera rays.

    RealSense pixel centres use integer coordinates; Isaac's pinhole helper
    uses index + 0.5. Nearest sampling avoids inventing depths across edges.
    The subsequent policy resize remains bilinear, as in sim2sim.
    """

    def __init__(self, intrinsics, *, min_coverage: float = 0.95):
        values = [intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy]
        if not np.isfinite(values).all() or min(intrinsics.fx, intrinsics.fy) <= 0:
            raise ValueError("Invalid native camera intrinsics")
        if intrinsics.width <= 0 or intrinsics.height <= 0:
            raise ValueError("Invalid native image dimensions")
        # D435i rectified depth normally has zero distortion. Refuse unhandled
        # distortion rather than silently treating a distorted image as pinhole.
        if not np.isfinite(intrinsics.coeffs).all() or np.any(np.abs(intrinsics.coeffs) > 1e-8):
            raise ValueError("Nonzero depth distortion: rectification is required before this remapper")
        self.native_shape = (intrinsics.height, intrinsics.width)
        target = TRAINING_INTRINSICS
        x = (np.arange(target.width) + 0.5 - target.cx) / target.fx
        y = (np.arange(target.height) + 0.5 - target.cy) / target.fy
        u, v = np.meshgrid(x * intrinsics.fx + intrinsics.ppx, y * intrinsics.fy + intrinsics.ppy)
        self.cols = np.floor(u + 0.5).astype(np.int32)
        self.rows = np.floor(v + 0.5).astype(np.int32)
        self.in_bounds = (
            (self.cols >= 0) & (self.cols < intrinsics.width) & (self.rows >= 0) & (self.rows < intrinsics.height)
        )
        # Only rays retained by the policy crop matter for coverage.
        self.coverage = float(self.in_bounds[:-2, 4:-4].mean())
        if self.coverage < min_coverage:
            raise ValueError(
                f"Native depth FOV covers only {self.coverage:.1%} of the policy crop; "
                f"need {min_coverage:.1%}. Check stream profile and calibration."
            )
        np.clip(self.cols, 0, intrinsics.width - 1, out=self.cols)
        np.clip(self.rows, 0, intrinsics.height - 1, out=self.rows)
        # Convert sampled Z using the actual native pixel ray, not RGB intrinsics.
        ray_x = (self.cols - intrinsics.ppx) / intrinsics.fx
        ray_y = (self.rows - intrinsics.ppy) / intrinsics.fy
        self.scale_map = np.sqrt(1.0 + ray_x**2 + ray_y**2).astype(np.float32)

    def convert(self, z16: np.ndarray, depth_scale: float) -> tuple[np.ndarray, float]:
        if z16.shape != self.native_shape:
            raise ValueError(f"Native depth shape changed: {z16.shape} != {self.native_shape}")
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("Depth scale must be finite and positive")
        sampled = np.asarray(z16[self.rows, self.cols], dtype=np.float32) * depth_scale
        sampled[~self.in_bounds] = 0.0
        valid_fraction = float((np.isfinite(sampled[:-2, 4:-4]) & (sampled[:-2, 4:-4] > 0)).mean())
        ranges = convert_image_plane_depth_to_camera_distance(sampled, scale_map=self.scale_map, max_distance=2.0)
        return ranges, valid_fraction


@dataclass(frozen=True)
class CameraSample:
    sequence: int
    received_at: float
    sensor_timestamp_ms: float
    valid_fraction: float
    image: np.ndarray


class RealSenseDepth:
    """Acquire on a worker so the motor loop never waits for USB frames.

    Freshness uses monotonic host arrival time and advancing device frame IDs.
    This detects lost/frozen delivery, not absolute sensor-to-host latency.
    """

    def __init__(self, *, serial=None, width=848, height=480, fps=60, min_valid_fraction=0.1):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._error = None
        self._thread = None
        self.min_valid_fraction = min_valid_fraction
        config = rs.config()
        if serial:
            config.enable_device(serial)
        elif len(rs.context().query_devices()) != 1:
            raise RuntimeError("Expected exactly one RealSense device; use --serial when multiple are connected")
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        try:
            sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = sensor.get_depth_scale()
            intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            self.remapper = DepthRemapper(intrinsics)
            print(f"Depth stream: {width}x{height}@{fps}, scale={self.depth_scale} m/unit")
            print(f"Native depth intrinsics: {intrinsics}; policy-ray coverage={self.remapper.coverage:.1%}")
            self._thread = threading.Thread(target=self._capture, name="d435i-depth", daemon=True)
            self._thread.start()
        except BaseException:
            self.pipeline.stop()
            raise

    def _capture(self):
        last_sequence = -1
        last_timestamp = -1.0
        try:
            while not self._stop.is_set():
                frames = self.pipeline.poll_for_frames()
                if not frames:
                    self._stop.wait(0.002)
                    continue
                frame = frames.get_depth_frame()
                if not frame:
                    continue
                received_at = time.monotonic()
                sequence = frame.get_frame_number()
                timestamp = frame.get_timestamp()
                if not np.isfinite(timestamp):
                    raise RuntimeError("Non-finite depth timestamp")
                if sequence <= last_sequence or timestamp <= last_timestamp:
                    continue
                last_sequence, last_timestamp = sequence, timestamp
                raw, fraction = self.remapper.convert(np.asanyarray(frame.get_data()), self.depth_scale)
                sample = CameraSample(sequence, received_at, timestamp, fraction, preprocess_range_image(raw))
                with self._lock:
                    self._latest = sample
        except Exception as exc:
            with self._lock:
                self._error = str(exc)

    def latest(self, max_age: float) -> CameraSample:
        with self._lock:
            sample, error = self._latest, self._error
        if error:
            raise RuntimeError(f"RealSense acquisition failed: {error}")
        if sample is None:
            raise RuntimeError("No depth frame yet")
        age = time.monotonic() - sample.received_at
        if age < 0 or age > max_age:
            raise RuntimeError(f"Depth frame stale: {age:.3f}s (limit {max_age:.3f}s)")
        if sample.valid_fraction < self.min_valid_fraction:
            raise RuntimeError(f"Only {sample.valid_fraction:.1%} of policy depth pixels have valid measurements")
        return sample

    def wait_ready(self, timeout=10.0, max_age=0.15):
        deadline = time.monotonic() + timeout
        reason = "No depth frame yet"
        while time.monotonic() < deadline:
            try:
                return self.latest(max_age)
            except RuntimeError as exc:
                reason = str(exc)
                time.sleep(0.02)
        raise RuntimeError(f"Camera did not become ready: {reason}")

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.pipeline.stop()


class StudentPolicy:
    """Strict feedforward ONNX contract: 45 proprio + 58x87 depth -> 12 actions."""

    def __init__(self, path, threads=1):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 5091] or inputs[0].type != "tensor(float)":
            raise ValueError("Expected one float32 ONNX input [1, 5091]; recurrent/other policies are unsupported")
        if len(outputs) != 1 or outputs[0].shape != [1, 12] or outputs[0].type != "tensor(float)":
            raise ValueError("Expected one float32 ONNX output [1, 12]")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def infer(self, proprio, depth):
        proprio = np.asarray(proprio, dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        if proprio.shape != (1, 45) or depth.shape != (58, 87):
            raise ValueError(f"Invalid observation shapes: {proprio.shape}, {depth.shape}")
        obs = np.concatenate((proprio, depth.reshape(1, -1)), axis=1)
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite policy observation")
        actions = self.session.run([self.output_name], {self.input_name: obs})[0]
        if actions.shape != (1, 12) or not np.isfinite(actions).all():
            raise RuntimeError("Invalid/non-finite policy action")
        # Training JointPositionAction uses this clip, before scale + offset.
        return np.clip(actions, -100.0, 100.0)


def preview_image(sample):
    return depth_preview_uint8(sample.image, normalized=True, max_distance=2.0)
