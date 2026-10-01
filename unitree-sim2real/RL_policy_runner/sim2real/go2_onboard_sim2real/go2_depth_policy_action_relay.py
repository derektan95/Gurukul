"""Run the Go2 depth policy and relay actions; optionally serve onboard USB depth.

This process reads the physical Go2's lowstate and wireless-controller input,
runs a 5091-D ONNX depth policy, and sends the resulting 12 raw policy
actions over UDP.

It deliberately never publishes rt/lowcmd and therefore does not command the
physical Go2. It is intended for hardware-in-the-loop action replay with a
Unitree MuJoCo bridge running on another computer.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# UDP packet format
# Must match go2_depth_action_receiver.py /
# go2_depth_student_action_relay.py.
# ---------------------------------------------------------------------------

PACKET_MAGIC = 0x47324441  # G2DA
PACKET_VERSION = 1
ACTION_PACKET = struct.Struct("!IHHQQ12f")


def pack_action(seq: int, timestamp_ns: int, actions: np.ndarray) -> bytes:
    actions = np.asarray(actions, dtype=np.float32).reshape(-1)

    if actions.shape != (12,) or not np.isfinite(actions).all():
        raise ValueError("Relay actions must be 12 finite float32 values")

    if seq < 0 or timestamp_ns < 0:
        raise ValueError("Sequence and timestamp must be nonnegative")

    return ACTION_PACKET.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        12,
        int(seq),
        int(timestamp_ns),
        *actions,
    )


# ---------------------------------------------------------------------------
# Unitree imports
# Subscriber only: there is intentionally no ChannelPublisher / LowCmd.
# ---------------------------------------------------------------------------

def _import_sdk():
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
        LowState_,
        WirelessController_,
    )

    return (
        ChannelFactoryInitialize,
        ChannelSubscriber,
        LowState_,
        WirelessController_,
    )


def quat_to_rot(quat):
    w, x, y, z = quat

    return np.asarray(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )


# Shared camera preprocessing lives at the unitree-sim2real root. The copied
# onboard camera helper is one directory deeper than the original helper.
for parent in Path(__file__).resolve().parents:
    if (parent / "shared_depth.py").is_file():
        sys.path.insert(0, str(parent))
        break

DEPTH_HEADER = struct.Struct("!4sQdd")  # magic, frame ID, host age, valid fraction
DEPTH_BYTES = 58 * 87 * 4


def recv_exact(connection, size, deadline):
    data = bytearray()
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Depth transfer timed out")
        connection.settimeout(remaining)
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise RuntimeError("Depth server disconnected")
        data.extend(chunk)
    return bytes(data)


class RemoteDepth:
    """Request the latest processed frame, with no streaming backlog.

    Server frame age plus request round-trip time bounds host-delivery age
    without synchronizing clocks. USB sensor latency is not measured.
    """

    def __init__(self, host, port):
        self.connection = socket.create_connection((host, port), timeout=5.0)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sequence = None
        self.advanced_at = None

    def latest(self, max_age):
        started = time.monotonic()
        deadline = started + max_age
        self.connection.settimeout(max_age)
        self.connection.sendall(b"D")
        header = recv_exact(self.connection, DEPTH_HEADER.size, deadline)
        magic, sequence, age, fraction = DEPTH_HEADER.unpack(header)
        if magic != b"G2D1":
            raise RuntimeError("Invalid depth header")
        payload = recv_exact(self.connection, DEPTH_BYTES, deadline)
        now = time.monotonic()
        if not np.isfinite([age, fraction]).all() or age < 0 or not 0.1 <= fraction <= 1:
            raise RuntimeError("Invalid depth metadata or insufficient valid depth")
        if age + now - started > max_age:
            raise RuntimeError("Remote depth frame stale")
        if self.sequence is not None and sequence < self.sequence:
            raise RuntimeError("Depth frame sequence went backwards")
        if sequence != self.sequence:
            self.sequence, self.advanced_at = sequence, now
        elif now - self.advanced_at > max_age:
            raise RuntimeError("Remote depth frame sequence frozen")
        image = np.frombuffer(payload, dtype="<f4").reshape(58, 87).copy()
        if not np.isfinite(image).all() or np.any((image < -0.5) | (image > 0.5)):
            raise RuntimeError("Invalid normalized depth image")
        return image

    def close(self):
        self.connection.close()


def serve_camera(args):
    # Only this mode opens the USB camera; no DDS or policy is needed onboard.
    from go2_depth_camera import RealSenseDepth

    with ExitStack() as resources:
        camera = RealSenseDepth(serial=args.serial, width=args.width, height=args.height, fps=args.fps)
        resources.callback(camera.close)
        camera.wait_ready(max_age=args.max_depth_age)
        server = resources.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.depth_bind, args.depth_port))
        server.listen(1)
        server.settimeout(0.5)
        deadline = time.monotonic() + args.duration if args.duration else None
        print(f"Depth camera server listening on {args.depth_bind}:{args.depth_port}", flush=True)
        while deadline is None or time.monotonic() < deadline:
            try:
                connection, address = server.accept()
            except socket.timeout:
                continue
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(0.5)
                print(f"Depth client connected: {address}", flush=True)
                try:
                    while deadline is None or time.monotonic() < deadline:
                        try:
                            request = connection.recv(1)
                        except socket.timeout:
                            continue
                        if not request:
                            break
                        if request != b"D":
                            raise RuntimeError("Invalid depth request")
                        sample = camera.latest(args.max_depth_age)
                        header = DEPTH_HEADER.pack(
                            b"G2D1", sample.sequence,
                            time.monotonic() - sample.received_at, sample.valid_fraction,
                        )
                        connection.sendall(header + np.asarray(sample.image, dtype="<f4").tobytes())
                except (OSError, RuntimeError) as exc:
                    print(f"Depth client stopped: {exc}", flush=True)


class ActionRelay:
    def __init__(self, args, actor, udp_socket, camera):
        self.args = args
        self.camera = camera
        self.actor = actor
        self.udp_socket = udp_socket

        self.actor_input_name = actor.get_inputs()[0].name
        self.actor_output_name = actor.get_outputs()[0].name

        self.low_state = None
        self.state_at = None
        self.tick = None

        # Same initial remote state as the working hardware runner:
        # zero command until remote events are received.
        self.remote = np.zeros(4, dtype=np.float32)
        self.keys = 0

        self.previous_action = np.zeros((1, 12), dtype=np.float32)

        self.gravity = np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float32,
        )

        # Same FR, FL, RR, RL order and defaults as
        # go2_hardware_pos_obs45.py.
        self.default_angles = np.array(
            [
                -0.1, 0.8, -1.5,   # FR
                 0.1, 0.8, -1.5,   # FL
                -0.1, 1.0, -1.5,   # RR
                 0.1, 1.0, -1.5,   # RL
            ],
            dtype=np.float32,
        )

        self.action_scales = np.full(12, 0.25, dtype=np.float32)
        self.action_scales[[0, 3, 6, 9]] = 0.125

        self.sequence = 0

        self.last_report_at = time.monotonic()
        self.steps_since_report = 0

    # ------------------------------------------------------------------
    # DDS callbacks
    # ------------------------------------------------------------------

    def state_callback(self, msg):
        # Ignore duplicate robot ticks.
        if self.tick == msg.tick:
            return

        self.tick = msg.tick
        self.low_state = msg
        self.state_at = time.monotonic()

    def remote_callback(self, msg):
        values = np.asarray(
            [msg.lx, msg.ly, msg.rx, msg.ry],
            dtype=np.float32,
        )

        if not np.isfinite(values).all():
            return

        self.remote = values
        self.keys = int(msg.keys)

    def stopped_by_remote(self):
        # R1 + L1.
        return bool(self.keys & 1) and bool(self.keys & 2)

    # ------------------------------------------------------------------
    # Same 45-D proprioception as go2_hardware_pos_obs45.py
    # ------------------------------------------------------------------

    def observation(self):
        if self.low_state is None or self.state_at is None:
            raise RuntimeError("No Go2 lowstate received")

        state_age = time.monotonic() - self.state_at

        if state_age > self.args.max_state_age:
            raise RuntimeError(
                f"Go2 lowstate is stale: {state_age:.3f}s"
            )

        if self.stopped_by_remote():
            raise RuntimeError("R1+L1 relay stop requested")

        state = self.low_state

        # IMU quaternion.
        quat = np.asarray(
            state.imu_state.quaternion,
            dtype=np.float32,
        )

        if (
            not np.isfinite(quat).all()
            or not 0.95 <= np.linalg.norm(quat) <= 1.05
        ):
            raise RuntimeError("Invalid Go2 IMU quaternion")

        # 3: angular velocity, same 0.25 scaling as baseline.
        ang_vel = np.asarray(
            state.imu_state.gyroscope[:3],
            dtype=np.float32,
        ) * 0.25

        # 3: projected gravity.
        projected_gravity = quat_to_rot(quat).T @ self.gravity

        # 3: velocity command.
        #
        # Baseline:
        #   lin_vel_x = j_ly
        #   lin_vel_y = j_lx
        #   ang_vel_z = -j_rx
        command = np.array(
            [
                self.remote[1],
                self.remote[0],
                -self.remote[2],
            ],
            dtype=np.float32,
        )

        command = np.clip(command, -1.0, 1.0)

        # 12 joint positions and velocities.
        positions = np.asarray(
            [motor.q for motor in state.motor_state[:12]],
            dtype=np.float32,
        )

        velocities = np.asarray(
            [motor.dq for motor in state.motor_state[:12]],
            dtype=np.float32,
        )

        if (
            not np.isfinite(positions).all()
            or not np.isfinite(velocities).all()
        ):
            raise RuntimeError("Non-finite Go2 motor state")

        # Same ordering as go2_hardware_pos_obs45.py:
        #
        #   3 angular velocity
        #   3 projected gravity
        #   3 command
        #   12 joint-position offsets
        #   12 scaled joint velocities
        #   12 previous action
        #
        # Total = 45.
        obs = np.concatenate(
            (
                ang_vel,
                projected_gravity,
                command,
                positions - self.default_angles,
                velocities * 0.05,
                self.previous_action[0],
            )
        ).reshape(1, 45)

        obs = obs.astype(np.float32, copy=False)

        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite policy observation")

        return obs

    # ------------------------------------------------------------------
    # Policy + UDP
    # ------------------------------------------------------------------

    def step(self):
        # Fetch depth first so proprioception is sampled after the network wait.
        depth = self.camera.latest(self.args.max_depth_age)
        obs = self.observation()
        policy_obs = np.concatenate((obs, depth.reshape(1, -1)), axis=1)

        actions = self.actor.run(
            [self.actor_output_name],
            {self.actor_input_name: policy_obs},
        )[0]

        actions = np.asarray(actions, dtype=np.float32)

        if actions.shape != (1, 12):
            raise RuntimeError(
                f"Expected policy action shape (1, 12), got {actions.shape}"
            )

        if not np.isfinite(actions).all():
            raise RuntimeError("Policy produced non-finite actions")

        raw_actions = actions[0]

        packet = pack_action(
            self.sequence,
            time.time_ns(),
            raw_actions,
        )

        if self.sequence % 50 == 0:
            print(
                f"TX seq={self.sequence} "
                f"actions={np.array2string(raw_actions, precision=9, max_line_width=1000)}",
                flush=True,
            )

        self.udp_socket.sendto(
            packet,
            self.args.destination,
        )

        self.previous_action[0] = raw_actions
        self.sequence += 1
        self.steps_since_report += 1

        # Human-readable report approximately once per second.
        now = time.monotonic()
        elapsed = now - self.last_report_at

        if elapsed >= 1.0:
            command = obs[0, 6:9]

            # Actual joint positions of the physical Go2.
            physical_q = np.asarray(
                [motor.q for motor in self.low_state.motor_state[:12]],
                dtype=np.float32,
            )

            # Exactly what the known-working hardware controller would command.
            q_target = (
                self.default_angles
                + raw_actions * self.action_scales
            )

            print(
                f"\n[{self.sequence:6d}] "
                f"rate={self.steps_since_report / elapsed:5.1f} Hz"
            )

            print(
                "command [vx vy wz]: "
                f"[{command[0]: .3f}, "
                f"{command[1]: .3f}, "
                f"{command[2]: .3f}]"
            )

            print(
                "physical q: "
                + np.array2string(
                    physical_q,
                    precision=3,
                    suppress_small=True,
                    floatmode="fixed",
                )
            )

            # print(
            #     "raw actions: "
            #     + np.array2string(
            #         raw_actions,
            #         precision=3,
            #         suppress_small=True,
            #         floatmode="fixed",
            #     )
            # )

            # print(
            #     "q_target:   "
            #     + np.array2string(
            #         q_target,
            #         precision=3,
            #         suppress_small=True,
            #         floatmode="fixed",
            #     )
            # )

            # print(
            #     f"action range: "
            #     f"[{raw_actions.min(): .3f}, {raw_actions.max(): .3f}], "
            #     f"mean |a|={np.mean(np.abs(raw_actions)):.3f}"
            # )

            self.last_report_at = now
            self.steps_since_report = 0

        return obs, raw_actions


def load_policy(path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1

    actor = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    inputs = actor.get_inputs()
    outputs = actor.get_outputs()

    if (
        len(inputs) != 1
        or inputs[0].shape != [1, 5091]
        or inputs[0].type != "tensor(float)"
    ):
        raise ValueError(
            "Depth policy requires one float32 input [1, 5091] (45 proprio + 58x87 depth)"
        )

    if (
        len(outputs) != 1
        or outputs[0].shape != [1, 12]
        or outputs[0].type != "tensor(float)"
    ):
        raise ValueError(
            "Locomotion policy requires one float32 output [1, 12]"
        )

    print(
        f"Loaded policy: {path}\n"
        f"  input:  {inputs[0].name} {inputs[0].shape}\n"
        f"  output: {outputs[0].name} {outputs[0].shape}"
    )

    return actor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--policy",
        type=Path,
        required=False,
    )

    parser.add_argument(
        "--interface",
        required=False,
        help="Go2 DDS interface; this program never publishes lowcmd",
    )

    parser.add_argument(
        "--pc-ip",
        required=False,
    )

    parser.add_argument(
        "--pc-port",
        type=int,
        default=15150,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-state-age",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="0 means run until Ctrl+C or R1+L1",
    )

    parser.add_argument("--camera-server", action="store_true", help="Serve USB depth on the onboard computer")
    parser.add_argument("--depth-host", help="Onboard computer IP serving the USB camera")
    parser.add_argument("--depth-port", type=int, default=15151)
    parser.add_argument("--depth-bind", default="0.0.0.0", help="Camera-server bind address")
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-depth-age", type=float, default=0.15)
    args = parser.parse_args(argv)
    if not 1 <= args.depth_port <= 65535:
        parser.error("Invalid depth port")
    if not np.isfinite(args.max_depth_age) or args.max_depth_age <= 0:
        parser.error("--max-depth-age must be finite and positive")
    if not np.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and nonnegative")
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("Camera dimensions and FPS must be positive")
    if args.camera_server:
        return args
    if not all((args.policy, args.interface, args.pc_ip, args.depth_host)):
        parser.error("Relay requires --policy, --interface, --pc-ip, and --depth-host")

    if not args.policy.is_file():
        parser.error(f"Policy not found: {args.policy}")

    if not 1 <= args.pc_port <= 65535:
        parser.error("Invalid PC port")

    if args.duration < 0:
        parser.error("--duration cannot be negative")

    if args.threads <= 0:
        parser.error("--threads must be positive")

    if (
        not np.isfinite(args.max_state_age)
        or args.max_state_age <= 0
    ):
        parser.error("--max-state-age must be finite and positive")

    args.destination = (
        args.pc_ip,
        args.pc_port,
    )

    return args


def main(argv=None):
    args = parse_args(argv)
    if args.camera_server:
        serve_camera(args)
        return
    with ExitStack() as resources:
        run_relay(args, resources)


def run_relay(args, resources):
    (
        ChannelFactoryInitialize,
        ChannelSubscriber,
        LowState_,
        WirelessController_,
    ) = _import_sdk()

    # Physical Go2 DDS.
    ChannelFactoryInitialize(
        0,
        args.interface,
    )

    actor = load_policy(
        args.policy,
        args.threads,
    )

    udp_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    resources.callback(udp_socket.close)
    camera = RemoteDepth(args.depth_host, args.depth_port)
    resources.callback(camera.close)

    relay = ActionRelay(
        args,
        actor,
        udp_socket,
        camera,
    )

    state_sub = ChannelSubscriber(
        "rt/lowstate",
        LowState_,
    )

    remote_sub = ChannelSubscriber(
        "rt/wirelesscontroller",
        WirelessController_,
    )

    state_sub.Init(
        relay.state_callback,
        10,
    )

    remote_sub.Init(
        relay.remote_callback,
        10,
    )

    # Avoid the startup race we encountered with the depth relay.
    print("Waiting for Go2 lowstate...")

    deadline_state = time.monotonic() + 5.0

    while relay.low_state is None:
        if time.monotonic() > deadline_state:
            raise RuntimeError(
                "Timed out waiting for Go2 lowstate"
            )

        time.sleep(0.01)

    print("Go2 lowstate received.")
    print(
        f"Relaying raw 12-D actions to "
        f"{args.pc_ip}:{args.pc_port} at 50 Hz."
    )
    print(
        "No rt/lowcmd publisher exists in this process; "
        "the physical Go2 is not commanded."
    )

    deadline = (
        time.monotonic() + args.duration
        if args.duration
        else None
    )

    try:
        while (
            deadline is None
            or time.monotonic() < deadline
        ):
            started = time.monotonic()

            relay.step()

            # Same policy rate as the baseline:
            # 200-Hz motor loop / decimation 4 = 50 Hz.
            time.sleep(
                max(
                    0.0,
                    0.02 - (time.monotonic() - started),
                )
            )

    finally:
        udp_socket.close()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "Relay stopped; no motor command "
            "was published to the physical Go2."
        )

    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Relay stopped: {exc}")
        raise SystemExit(1) from exc
