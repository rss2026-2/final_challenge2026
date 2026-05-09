#!/usr/bin/env python3

"""
Plot steering angle and steering rate from one or more rosbag topics over time.

Usage:
    python3 steering_angle_viz.py /path/to/bag1 /path/to/bag2
    python3 steering_angle_viz.py /path/to/bag --topic /vesc/low_level/input/navigation
    python3 steering_angle_viz.py /path/to/bag1 /path/to/bag2 --output steering_plot.png
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py

from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Configure per-bag alignment windows here.
# The tuple is (start_s, end_s) measured from the start of that bag.
# Leave a bag out of this dict to use the full bag from t=0 onward.
BAG_WINDOWS_S: Dict[str, Tuple[float, Optional[float]]] = {
    "bags/rosbag2_2026_05_03-22_54_54/": (21.0, 83.0),
    # "my_trial_bag": (3.5, 18.0),
}


def resolve_bag_uri(bag_path: str) -> str:
    bag_path = os.path.expanduser(bag_path)
    if bag_path.endswith(".db3"):
        return os.path.dirname(bag_path)
    return bag_path


def bag_name_from_path(bag_path: str) -> str:
    bag_path = os.path.expanduser(bag_path)
    if bag_path.endswith(".db3"):
        bag_path = os.path.dirname(bag_path)
    return os.path.basename(os.path.normpath(bag_path))


def get_bag_window(bag_path: str) -> Tuple[float, Optional[float]]:
    normalized_path = os.path.normpath(os.path.expanduser(bag_path))
    bag_name = bag_name_from_path(bag_path)

    for key in (bag_name, normalized_path, os.path.expanduser(bag_path)):
        if key in BAG_WINDOWS_S:
            return BAG_WINDOWS_S[key]

    return 0.0, None


def read_steering_from_bag(
    bag_path: str,
    topic: str,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    bag_uri = resolve_bag_uri(bag_path)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_uri, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    msg_type = None
    for t in reader.get_all_topics_and_types():
        if t.name == topic:
            msg_type = get_message(t.type)
            break

    if msg_type is None:
        raise ValueError(f"Topic '{topic}' was not found in bag '{bag_path}'")

    times_s: List[float] = []
    steering_angles: List[float] = []
    bag_start_ns = None

    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()
        if topic_name != topic:
            continue

        if bag_start_ns is None:
            bag_start_ns = timestamp_ns

        msg = deserialize_message(data, msg_type)
        if not isinstance(msg, AckermannDriveStamped):
            raise TypeError(
                f"Topic '{topic}' has type '{msg_type.__name__}', expected AckermannDriveStamped"
            )

        t_s = (timestamp_ns - bag_start_ns) * 1e-9
        if t_s < start_s:
            continue
        if end_s is not None and t_s > end_s:
            break

        times_s.append(t_s - start_s)
        steering_angles.append(float(msg.drive.steering_angle))

    return times_s, steering_angles


def steering_rate(times_s: List[float], values: List[float]) -> List[float]:
    if not times_s:
        return []

    if len(times_s) == 1:
        return [0.0]

    times_np = np.asarray(times_s, dtype=np.float64)
    values_np = np.asarray(values, dtype=np.float64)
    return list(np.gradient(values_np, times_np))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot steering angle and steering rate from one or more rosbags."
    )
    parser.add_argument(
        "bag_paths",
        nargs="+",
        help="One or more rosbag2 directories or .db3 files",
    )
    parser.add_argument(
        "--topic",
        default="/vesc/low_level/input/navigation",
        help="AckermannDriveStamped topic to read",
    )
    parser.add_argument(
        "--output",
        default="steering_angle_analysis.png",
        help="Output image path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving",
    )
    args = parser.parse_args()

    fig, (ax_angle, ax_rate) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    fig.suptitle("Steering Angle and Steering Rate")

    for trial_idx, bag_path in enumerate(args.bag_paths, start=1):
        bag_name = bag_name_from_path(bag_path)
        print(bag_name)
        start_s, end_s = get_bag_window(bag_path)
        print(f"  window: start={start_s:.2f}s, end={'None' if end_s is None else f'{end_s:.2f}s'}")
        times_s, steering_angles = read_steering_from_bag(
            bag_path,
            args.topic,
            start_s=start_s,
            end_s=end_s,
        )
        if not times_s:
            raise RuntimeError(f"No messages found on topic '{args.topic}' in bag '{bag_name}'")

        rates = steering_rate(times_s, steering_angles)
        label = f"Trial {trial_idx}"
        if end_s is None:
            window_label = f"start={start_s:.2f}s"
        else:
            window_label = f"{start_s:.2f}s to {end_s:.2f}s"

        ax_angle.plot(times_s, steering_angles, linewidth=1.5, label=f"{label} ({window_label})")
        ax_rate.plot(times_s, rates, linewidth=1.5, label=f"{label} ({window_label})")

    ax_angle.set_ylabel("Steering angle (rad)")
    ax_angle.grid(True, alpha=0.3)
    ax_angle.axhline(0.0, color="black", linewidth=2.5, linestyle="-", alpha=0.8)
    ax_angle.legend(loc="best")

    ax_rate.set_xlabel("Time since aligned start (s)")
    ax_rate.set_ylabel("Steering rate (rad/s)")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.legend(loc="best")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
