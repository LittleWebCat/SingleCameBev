#!/usr/bin/env python3
"""Stream camera-only bag images on a constant-rate video clock.

The iterator reads only the image topic, decodes only selected images, and holds
the last saved image across capture gaps. It never starts ROS nodes or reads
LiDAR messages. Returned image arrays are read-only; copy before drawing.
"""

from pathlib import Path
import math
import struct

import cv2
import numpy as np
import yaml


CAMERA_TOPIC = "/camera/front/image_raw"


def camera_bag_timeline(bag):
    """Return the saved bag clock without scanning messages."""
    bag = Path(bag).resolve()
    metadata = yaml.safe_load((bag / "metadata.yaml").read_text())[
        "rosbag2_bagfile_information"
    ]
    start = int(metadata["starting_time"]["nanoseconds_since_epoch"])
    duration = int(metadata["duration"]["nanoseconds"])
    return {
        "source_bag": str(bag),
        "storage_id": metadata["storage_identifier"],
        "bag_start_ns": start,
        "duration_ns": duration,
        "end_ns": start + duration,
        "timestamp_reference": "video time zero is the first saved camera receipt",
    }


def _decode_image(serialized):
    """Decode the standard CDR Image layout without copying the entire message."""
    data = memoryview(serialized)
    encapsulation = struct.unpack_from(">H", data, 0)[0]
    if encapsulation not in (0, 1):
        raise ValueError("Camera image must use standard CDR1 serialization")
    endian = "<" if encapsulation == 1 else ">"
    position = 4

    def scalar(code):
        nonlocal position
        size = struct.calcsize(code)
        position += (-(position - 4)) % size
        value = struct.unpack_from(endian + code, data, position)[0]
        position += size
        return value

    def string():
        nonlocal position
        length = scalar("I")
        value = bytes(data[position:position + length - 1]).decode("utf-8")
        position += length
        return value

    seconds, nanoseconds = scalar("i"), scalar("I")
    frame_id = string()
    height, width, encoding = scalar("I"), scalar("I"), string()
    scalar("B")  # Byte endian flag has no effect on three-channel 8-bit images.
    step, length = scalar("I"), scalar("I")
    if encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"Unsupported camera image encoding: {encoding}")
    if step < width * 3 or length != height * step or position + length > len(data):
        raise ValueError("Cannot decode saved camera image payload layout")
    rows = np.frombuffer(data[position:position + length], np.uint8).reshape(height, step)
    image = rows[:, :width * 3].reshape(height, width, 3)
    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    image.setflags(write=False)
    return image, {
        "source_stamp_ns": seconds * 1_000_000_000 + nanoseconds,
        "frame_id": frame_id,
        "width": width,
        "height": height,
        "source_encoding": encoding,
    }


def iter_camera_frames(bag, fps=10.0):
    """Yield ``(bgr_image, metadata)`` at regular video times from a saved bag.

    A video slot selects the latest camera receipt at or before that slot.
    ``held`` is true only when the previous slot used the same saved image.
    Prediction renderers can reuse previous predictions for held frames.
    Both receipt and source-header timestamps remain available; neither is
    claimed to be an exposure synchronization measurement.
    """
    import rosbag2_py

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    timeline = camera_bag_timeline(bag)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=timeline["source_bag"], storage_id=timeline["storage_id"]
        ),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[CAMERA_TOPIC]))

    def next_saved_image():
        if reader.has_next():
            _, serialized, receipt = reader.read_next()
            return serialized, int(receipt)
        return None

    try:
        current = next_saved_image()
        if current is None:
            raise ValueError(f"Bag has no images on {CAMERA_TOPIC}")
        video_start = current[1]
        upcoming = next_saved_image()
        current_image, source = _decode_image(current[0])
        decoded_receipt = current[1]
        previous_receipt = None
        index = 0
        while True:
            target = video_start + round(index * 1_000_000_000 / fps)
            if target > timeline["end_ns"]:
                break
            while upcoming is not None and upcoming[1] <= target:
                current = upcoming
                upcoming = next_saved_image()
            if current[1] != decoded_receipt:
                current_image, source = _decode_image(current[0])
                decoded_receipt = current[1]
            yield current_image, {
                **source,
                "frame_index": index,
                "time_s": index / fps,
                "bag_offset_s": (target - timeline["bag_start_ns"]) / 1e9,
                "target_recorded_stamp_ns": target,
                "source_age_s": (target - current[1]) / 1e9,
                "recorded_stamp_ns": current[1],
                "receipt_ns": current[1],
                "held": current[1] == previous_receipt,
                "timestamp_reference": timeline["timestamp_reference"],
            }
            previous_receipt = current[1]
            index += 1
    finally:
        reader.close()
