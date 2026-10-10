#!/usr/bin/python3
"""Extract a small indexed, native camera/LiDAR sample set for range inspection.

Uses recorded receipt times to seek short windows, then pairs by original source
headers. It never starts sensor drivers or performs model inference. All finite
raw XYZ returns are retained, including zero ranges, with no ego/range/display
crop. Factory LiDAR-to-sensor translation is converted from millimetres to metres.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rosbag2_py
import yaml
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud2

from combined_bev_display import load_calibration
from combined_bev_live import header_stamp


def decode_raw_cloud(message, transform):
    """Read native organized strides using the same field rules as the viewer."""
    types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    fields = {f.name: f for f in message.fields}
    if message.header.frame_id not in ("os_lidar", "os_sensor"):
        raise ValueError(f"Unexpected cloud frame {message.header.frame_id}")
    if message.row_step < message.width * message.point_step or len(message.data) < message.row_step * message.height:
        raise ValueError("Incomplete PointCloud2 data")
    endian = ">" if message.is_bigendian else "<"

    def column(name):
        field = fields[name]
        if field.count != 1 or field.datatype not in types:
            raise ValueError(f"Unsupported cloud field {name}")
        dtype = np.dtype(endian + types[field.datatype])
        return np.ndarray((message.height, message.width), dtype=dtype,
                          buffer=message.data, offset=field.offset,
                          strides=(message.row_step, message.point_step)).reshape(-1)

    xyz = np.column_stack([column(k) for k in ("x", "y", "z")]).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    raw_range = np.linalg.norm(xyz[finite], axis=1)
    intensity_name = "reflectivity" if "reflectivity" in fields else "intensity"
    intensity = column(intensity_name).astype(np.float32) if intensity_name in fields else np.zeros(len(xyz), np.float32)
    if intensity_name == "reflectivity":
        intensity /= 255.
    xyz, intensity = xyz[finite], intensity[finite]
    if message.header.frame_id == "os_lidar":
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    return np.column_stack([xyz, intensity]).astype(np.float32), {
        "native_slots": int(message.width * message.height),
        "finite_raw_points": int(finite.sum()),
        "nonzero_raw_points": int(np.count_nonzero(raw_range >= .5)),
        "max_raw_range_m": float(raw_range.max(initial=0)),
        "intensity_field": intensity_name,
        "native_width": int(message.width), "native_height": int(message.height),
    }


def reader_for(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/camera/front/image_raw", "/points"]))
    return reader


def pair_at(bag, files, target_ns, half_window_ns=250_000_000):
    low, high = target_ns - half_window_ns, target_ns + half_window_ns
    images, clouds = [], []
    for part in files:
        start = part["starting_time"]["nanoseconds_since_epoch"]
        end = start + part["duration"]["nanoseconds"]
        if end < low or start > high:
            continue
        reader = reader_for(bag / part["path"])
        reader.seek(max(start, low))
        while reader.has_next():
            topic, payload, receipt = reader.read_next()
            if receipt > high:
                break
            if receipt < low:
                continue
            cls = Image if topic == "/camera/front/image_raw" else PointCloud2
            message = deserialize_message(payload, cls)
            item = (message, int(receipt), header_stamp(message.header))
            (images if cls is Image else clouds).append(item)
        del reader
    if not images or not clouds:
        raise RuntimeError(f"No paired messages around receipt {target_ns}")
    image = min(images, key=lambda x: abs(x[1] - target_ns))
    cloud = min(clouds, key=lambda x: abs(x[2] - image[2]))
    return image, cloud


def contact_sheet(samples, output):
    columns, tile_width, tile_height = 4, 384, 312
    sheet = np.full(((len(samples) + columns - 1) // columns * tile_height,
                     columns * tile_width, 3), 18, np.uint8)
    for i, sample in enumerate(samples):
        tile = cv2.imread(str(output / sample["image"]))
        tile = cv2.resize(tile, (384, 288), interpolation=cv2.INTER_AREA)
        y, x = i // columns * tile_height, i % columns * tile_width
        sheet[y + 24:y + 312, x:x + 384] = tile
        label = f'{sample["sample_id"]}  t={sample["elapsed_s"]:.1f}s  pair={sample["pair_delta_ms"]:+.1f}ms'
        cv2.putText(sheet, label, (x + 6, y + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / "contactsheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/lidar_range_10hz_001"))
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    bag, output = args.bag.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    output.mkdir(parents=True)
    capture = Path(str(bag) + ".capture")
    info = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    metadata = json.loads((capture / "ouster_metadata.json").read_text())
    transform = np.asarray(metadata["lidar_intrinsics"]["lidar_to_sensor_transform"], dtype=np.float64).reshape(4, 4)
    transform[:3, 3] /= 1000.
    calibration = load_calibration(capture / "front_camera_1024x768.candidate.yaml", capture / "T_B_os_sensor.candidate.yaml")
    first = int(info["starting_time"]["nanoseconds_since_epoch"])
    duration = int(info["duration"]["nanoseconds"])
    margins = min(1_000_000_000, duration // 10)
    offsets = np.linspace(margins, duration - margins, args.samples).astype(np.int64)
    manifest = {"bag": str(bag), "bag_first_receipt_ns": first,
                "bag_duration_s": duration / 1e9, "lidar_mode": metadata["config_params"]["lidar_mode"],
                "cloud_frame": "os_sensor", "cloud_dtype": "float32", "cloud_columns": ["x", "y", "z", "reflectivity_normalized"],
                "point_filter": "finite raw XYZ only; zero ranges retained; no ego/range/display crop",
                "factory_sensor_from_lidar_m": transform.tolist(), "calibration": calibration,
                "calibration_status": "Imported candidate B; not validated ground truth",
                "pairing": "nearest camera receipt to each target; nearest cloud source header within indexed +/-250ms receipt window",
                "samples": []}
    bridge = CvBridge()
    started = time.monotonic()
    for i, offset in enumerate(offsets):
        image, cloud = pair_at(bag, info["files"], first + int(offset))
        pixels = bridge.imgmsg_to_cv2(image[0], desired_encoding="bgr8")
        points, cloud_info = decode_raw_cloud(cloud[0], transform)
        image_name, cloud_name = f"{i:03d}.png", f"{i:03d}.xyzi.bin"
        cv2.imwrite(str(output / image_name), pixels)
        points.tofile(output / cloud_name)
        sample = {"sample_id": f"{i:03d}", "image": image_name, "cloud": cloud_name,
                  "image_stamp_ns": image[2], "lidar_stamp_ns": cloud[2],
                  "image_receipt_ns": image[1], "lidar_receipt_ns": cloud[1],
                  "elapsed_s": (image[1] - first) / 1e9,
                  "pair_delta_ms": (cloud[2] - image[2]) / 1e6,
                  "image_frame_id": image[0].header.frame_id,
                  "lidar_native_frame_id": cloud[0].header.frame_id,
                  "point_count": len(points), **cloud_info}
        manifest["samples"].append(sample)
        print(f"Saved {sample['sample_id']} elapsed={sample['elapsed_s']:.2f}s points={len(points)} pair={sample['pair_delta_ms']:+.1f}ms", flush=True)
    manifest["extraction_wall_s"] = time.monotonic() - started
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    contact_sheet(manifest["samples"], output)
    print(f"Contact sheet: {output / 'contactsheet.jpg'}", flush=True)


if __name__ == "__main__":
    main()
