#!/usr/bin/env python3
"""Check the stationary C++ ground BEV on synthetic scenes and a saved RGB scan.

The child node and this verifier use private localhost topics in a separate ROS
domain. No camera, LiDAR, or RViz is started. Synthetic scenes have known planes,
obstacle heights, and occlusion; a real fixture verifies software conservation
and format only, not physical ground or calibration accuracy.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def cloud_metadata(message):
    return {"width": int(message.width), "height": int(message.height),
            "point_step": int(message.point_step), "row_step": int(message.row_step),
            "is_bigendian": bool(message.is_bigendian), "is_dense": bool(message.is_dense),
            "fields": [{"name": item.name, "offset": int(item.offset),
                        "datatype": int(item.datatype), "count": int(item.count)}
                       for item in message.fields]}


def scalar_field(metadata, payload, name, expected_datatype=None):
    """Read scalar fields without converting packed FLOAT32 RGB bits to floats."""
    import numpy as np
    fields = {item["name"]: item for item in metadata["fields"]}
    if name not in fields:
        raise ValueError(f"Cloud is missing {name}")
    item = fields[name]
    types = {2: ("u1", 1), 6: ("u4", 4), 7: ("f4", 4), 8: ("f8", 8)}
    if item["count"] != 1 or item["datatype"] not in types:
        raise ValueError(f"Cloud {name} must be a supported scalar")
    if expected_datatype is not None and item["datatype"] != expected_datatype:
        raise ValueError(f"Cloud {name} has datatype {item['datatype']}, expected {expected_datatype}")
    code, size = types[item["datatype"]]
    if name == "rgb":
        if item["datatype"] not in (6, 7):
            raise ValueError("Packed rgb must be FLOAT32 or UINT32")
        code = "u4"
    if item["offset"] < 0 or item["offset"] + size > metadata["point_step"]:
        raise ValueError(f"Cloud {name} extends beyond point_step")
    if metadata["row_step"] < metadata["width"] * metadata["point_step"]:
        raise ValueError("Cloud row_step is shorter than its packed row")
    if len(payload) != metadata["height"] * metadata["row_step"]:
        raise ValueError("Cloud data disagrees with its layout")
    if not metadata["width"] * metadata["height"]:
        return np.empty(0, dtype=code)
    order = ">" if metadata["is_bigendian"] else "<"
    return np.ndarray((metadata["height"], metadata["width"]), dtype=order + code,
                      buffer=payload, offset=item["offset"],
                      strides=(metadata["row_step"], metadata["point_step"])).reshape(-1).copy()


def synthetic_scene(slope=False, no_floor=False):
    """Independent truth: broad floor, a box front, and a perpendicular corner."""
    import numpy as np
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("rgb", "<u4"), ("source_index", "<u4"),
                      ("camera_colored", "u1"), ("padding", "u1", (3,))])
    a, b, c = (0.04, 0.03, -1.8) if slope else (0.0, 0.0, -1.8)
    if no_floor:
        y, z = np.meshgrid(np.linspace(-4, 4, 41), np.linspace(0.2, 1.8, 21))
        xyz = np.column_stack((np.full(y.size, 4.0), y.ravel(), z.ravel()))
        truth = np.ones(len(xyz), dtype=np.uint8)
    else:
        x, y = np.meshgrid(np.arange(1.1, 12.0, 0.2), np.arange(-4.9, 5.0, 0.2))
        ground = np.column_stack((x.ravel(), y.ravel(), (a*x + b*y + c).ravel()))
        # Box front at x=3.1; explicit ground returns behind it must remain unknown.
        oy, oh = np.meshgrid(np.arange(-0.5, 0.6, 0.2), np.arange(0.3, 1.1, 0.2))
        ox = np.full(oy.size, 3.1)
        box = np.column_stack((ox, oy.ravel(), a*ox + b*oy.ravel() + c + oh.ravel()))
        # A perpendicular side makes a corner and exercises noncentral map cells.
        cx, ch = np.meshgrid(np.arange(5.1, 6.2, 0.2), np.arange(0.3, 1.1, 0.2))
        cy = np.full(cx.size, 2.1)
        corner = np.column_stack((cx.ravel(), cy, a*cx.ravel() + b*cy + c + ch.ravel()))
        xyz = np.vstack((ground, box, corner, [[65.0, 0.0, -1.8]],
                         [[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0], [float("inf"), 1.0, 0.0]]))
        truth = np.concatenate((np.full(len(ground), 2, dtype=np.uint8),
                                np.full(len(box) + len(corner), 3, dtype=np.uint8),
                                np.array([1, 0, 0, 0], dtype=np.uint8)))
    points = np.zeros(len(xyz), dtype=dtype)
    for column, name in enumerate(("x", "y", "z")):
        points[name] = xyz[:, column]
    points["rgb"] = 0x123456
    points["source_index"] = 100000 + 3*np.arange(len(points), dtype=np.uint32)
    points["camera_colored"] = 1
    metadata = {"width": len(points), "height": 1, "point_step": 24,
                "row_step": len(points)*24, "is_bigendian": False, "is_dense": no_floor,
                "fields": [{"name": name, "offset": offset, "datatype": datatype, "count": 1}
                           for name, offset, datatype in (("x", 0, 7), ("y", 4, 7), ("z", 8, 7),
                                                         ("rgb", 12, 7), ("source_index", 16, 6),
                                                         ("camera_colored", 20, 2))]}
    return {"metadata": metadata, "data": points.tobytes(), "xyz": xyz.astype(np.float32).astype(np.float64),
            "source_indices": points["source_index"].copy(), "truth": truth,
            "plane": [a, b, c], "kind": "no_floor" if no_floor else "slope" if slope else "flat"}


VEHICLE_MASK = {"self_mask_enabled": True,
                "self_x_min_m": -2.131, "self_x_max_m": 2.350,
                "self_y_min_m": -0.905, "self_y_max_m": 0.885,
                "self_z_min_m": -1.8, "self_z_max_m": 0.0}


def vehicle_mask_scene():
    """Known body returns versus nearby obstacles, using the supplied body size.

    Explicit point labels establish the truth without repeating the node's AABB
    calculation. Edge pairs are 10 mm inside/outside to avoid FLOAT32 rounding
    ambiguity at decimal bounds. The background floor stays outside the body.
    """
    import numpy as np
    scene = synthetic_scene()
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("rgb", "<u4"), ("source_index", "<u4"),
                      ("camera_colored", "u1"), ("padding", "u1", (3,))])
    original = np.frombuffer(scene["data"], dtype=dtype)
    # Remove simulated floor under the body; everything kept is outside it.
    keep = (scene["truth"] != 2) | (original["x"] >= 2.5) | (np.abs(original["y"]) >= 1.1)
    known = [([1.0, 0.2, -0.6], 4),
             ([2.340, 0.0, -0.6], 4), ([2.360, 0.0, -0.6], 3),
             ([-2.121, 0.0, -0.6], 4), ([-2.141, 0.0, -0.6], 1),
             ([1.4, 0.875, -0.6], 4), ([1.4, 0.895, -0.6], 3),
             ([1.4, -0.895, -0.6], 4), ([1.4, -0.915, -0.6], 3),
             ([1.4, 0.2, -0.010], 4), ([1.4, 0.2, 0.010], 3),
             ([1.4, 0.2, -1.790], 4), ([1.4, 0.2, -1.810], 2)]
    added = np.zeros(len(known), dtype=dtype)
    for row, (xyz, unused_label) in enumerate(known):
        for name, value in zip(("x", "y", "z"), xyz):
            added[name][row] = value
    added["rgb"] = 0x123456
    points = np.concatenate((original[keep], added))
    points["source_index"] = 200000 + 7*np.arange(len(points), dtype=np.uint32)
    truth = np.concatenate((scene["truth"][keep], np.array([label for unused_xyz, label in known], dtype=np.uint8)))
    xyz = np.column_stack([points[name] for name in ("x", "y", "z")]).astype(np.float64)
    metadata = dict(scene["metadata"], width=len(points), row_step=len(points)*24)
    return dict(scene, metadata=metadata, data=points.tobytes(), xyz=xyz,
                source_indices=points["source_index"].copy(), truth=truth,
                kind="vehicle_mask", explicit_edge_start=len(points)-len(known))


@contextmanager
def child_process(arguments, report):
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            log.seek(0)
            report["node_log_tail"] = log.read().decode("utf-8", errors="replace")[-5000:]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path,
                        help="optional stationary RGB fixture containing input_points_rgb.bin and input_metadata.json")
    parser.add_argument("--executable", type=Path,
                        default=root / "install/bev_geometry/lib/bev_geometry/ground_bev_node")
    parser.add_argument("--output", type=Path, default=root / "evaluation/ground_bev_replay.json")
    parser.add_argument("--domain-id", type=int, default=43)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--check-vehicle-mask", action="store_true",
                        help="also run a second child with the supplied 4.481 x 1.790 m body bounds")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new evidence path")
    if not 2 <= args.domain_id <= 231:
        parser.error("domain-id must be between 2 and 231, separate from live domain 1")
    if not math.isfinite(args.timeout) or not 15 <= args.timeout <= 30:
        parser.error("timeout must be between 15 and 30 seconds")
    if not args.executable.is_file():
        parser.error(f"C++ executable has not been built: {args.executable}")

    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    os.environ.pop("ROS_STATIC_PEERS", None)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    os.environ.setdefault("CYCLONEDDS_URI", "file://" + str(root / "config/transport/cyclonedds_large_clouds.xml"))
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/camera_lidar_bev-ground-check-logs")

    import numpy as np
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from extrinsic_capture import decode_point_cloud

    report = {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "stationary_ground_software_integration_check", "ros_domain_id": args.domain_id,
              "live_hardware_used": False, "physical_ground_accuracy_validated": False,
              "physical_calibration_accuracy_validated": False, "timing_and_motion_validated": False,
              "source_files": {}, "verified_by_stage": {}, "stage_results": {}, "errors": [],
              "output_messages": 0, "clear_cloud_messages": 0, "clear_grid_messages": 0,
              "clear_image_messages": 0, "verified_image_messages": 0,
              "provisional_diagnostics": 0, "published_input_messages": 0,
              "mask_diagnostics_verified": 0}
    started = time.monotonic()
    deadline = started + args.timeout
    node = None
    output_records = {}
    published = {}
    verified = set()
    last_output = {}
    active_stage = "discovery"
    diagnostic_by_stage = {}
    stale_grid_stamps = set()
    stale_image_stamps = set()
    try:
        scenes = {"flat": synthetic_scene(), "slope": synthetic_scene(slope=True),
                  "no_floor": synthetic_scene(no_floor=True)}
        if args.check_vehicle_mask:
            scenes["vehicle_mask"] = vehicle_mask_scene()
        if args.fixture:
            paths = [args.fixture / "input_points_rgb.bin", args.fixture / "input_metadata.json"]
            for path in paths:
                report["source_files"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata = json.loads(paths[1].read_text())
            payload = paths[0].read_bytes()
            cloud = decode_point_cloud(metadata, payload)
            indices = (scalar_field(metadata, payload, "source_index", 6)
                       if any(item["name"] == "source_index" for item in metadata["fields"])
                       else np.arange(len(cloud.xyz), dtype=np.uint32))
            header = metadata.get("header", {})
            if header.get("frame_id") != "os_sensor":
                raise ValueError("Real fixture must explicitly identify os_sensor XYZ")
            if len(np.unique(indices)) != len(indices):
                raise ValueError("Real fixture source_index values must be unique")
            scenes["fixture"] = {"metadata": metadata, "data": payload, "xyz": cloud.xyz,
                                 "source_indices": indices, "kind": "fixture"}
            report["fixture"] = {"input_point_count": len(cloud.xyz), "original_header": header,
                                 "replay_header_policy": "fresh_host_stamps_stationary_geometry_only",
                                 "physical_accuracy_claim": "unverified; no measured box or floor height"}

        rclpy.init()
        node = rclpy.create_node("ground_bev_isolated_check")
        prefix = "/ground_bev_check"
        topics = {"input_topic": prefix + "/input", "ground_topic": prefix + "/ground",
                  "obstacle_topic": prefix + "/obstacles", "unknown_topic": prefix + "/unknown",
                  "classified_topic": prefix + "/classified", "occupancy_topic": prefix + "/occupancy",
                  "occupancy_image_topic": prefix + "/occupancy_image",
                  "diagnostics_topic": prefix + "/diagnostics"}
        reliable = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        publisher = node.create_publisher(PointCloud2, topics["input_topic"], reliable)
        label_semantics = {"ground": (2, 0x00ff00), "obstacle": (3, 0xff3030), "unknown": (1, 0x999999)}

        def record_error(error):
            if str(error) not in report["errors"]:
                report["errors"].append(str(error))

        def check_result(key):
            record = output_records[key]
            if key in verified or set(record) != {"ground", "obstacle", "unknown", "classified", "grid", "image"}:
                return
            item = published[key]
            stage, scene = item["stage"], item["scene"]
            if stage.startswith("wrong"):
                if any(len(record[name]["indices"]) for name in (*label_semantics, "classified")):
                    raise ValueError("Wrong-frame input produced classified points")
                if np.any(record["grid"]["data"] != -1):
                    raise ValueError("Wrong-frame input produced a known occupancy cell")
                return
            source_xyz = scene["xyz"]
            source_indices = scene["source_indices"]
            valid = np.isfinite(source_xyz).all(axis=1) & np.any(source_xyz != 0, axis=1)
            masked = (scene["truth"] == 4 if scene["kind"] == "vehicle_mask"
                      else np.zeros(len(source_xyz), dtype=bool))
            included = valid & ~masked
            expected = source_indices[included]
            output_indices = np.concatenate([record[name]["indices"] for name in label_semantics])
            if len(output_indices) != len(expected) or not np.array_equal(np.sort(output_indices), np.sort(expected)):
                raise ValueError(f"{stage}: finite nonzero points were lost, duplicated, or assigned invented indices")
            lookup = {int(index): offset for offset, index in enumerate(source_indices)}
            predicted = np.zeros(len(source_xyz), dtype=np.uint8)
            for name, (classification, unused_color) in label_semantics.items():
                output = record[name]
                offsets = np.array([lookup[int(index)] for index in output["indices"]], dtype=np.int64)
                predicted[offsets] = classification
                if not np.allclose(output["xyz"], source_xyz[offsets], atol=1e-5, rtol=1e-6):
                    raise ValueError(f"{stage}: {name} XYZ changed from the input sensor frame")
                if scene["kind"] in ("flat", "slope", "vehicle_mask") and name != "unknown":
                    a, b, c = scene["plane"]
                    heights = output["xyz"][:, 2] - (a*output["xyz"][:, 0] + b*output["xyz"][:, 1] + c)
                    # Allow either vertical or normal signed height for a sloped plane.
                    vertical_ok = np.allclose(output["height"], heights, atol=0.025, rtol=0)
                    normal_ok = np.allclose(output["height"], heights / math.sqrt(1+a*a+b*b), atol=0.025, rtol=0)
                    if not (vertical_ok or normal_ok):
                        raise ValueError(f"{stage}: height_above_ground disagrees with the known synthetic plane")
            combined = record["classified"]
            if (not np.array_equal(combined["indices"], expected)
                    or not np.allclose(combined["xyz"], source_xyz[included], atol=1e-5, rtol=1e-6)
                    or not np.array_equal(combined["labels"], predicted[included])):
                raise ValueError(f"{stage}: combined classified points differ from the three class clouds or original order")
            grid = record["grid"]
            # ROS grid has lateral rows and forward columns. The camera-style
            # BEV raster has forward rows (up) and lateral columns (left).
            oriented_cells = grid["data"].T[::-1, ::-1]
            expected_image = np.full((200, 150, 3), 48, dtype=np.uint8)
            expected_image[oriented_cells == 0] = [50, 170, 70]
            expected_image[oriented_cells == 100] = [230, 60, 50]
            if not np.array_equal(record["image"], expected_image):
                raise ValueError(f"{stage}: occupancy image orientation or RGB values disagree with nav grid")
            report["verified_image_messages"] += 1
            counts = {name: len(record[name]["indices"]) for name in label_semantics}
            outcome = {"input_points": len(source_xyz), "valid_points": int(np.count_nonzero(valid)),
                       "excluded_nonfinite_or_zero_points": int(np.count_nonzero(~valid)),
                       "excluded_known_body_points": int(np.count_nonzero(masked)),
                       "classes": counts,
                       "grid_counts": {"occupied": int(np.count_nonzero(grid["data"] == 100)),
                                       "free": int(np.count_nonzero(grid["data"] == 0)),
                                       "unknown": int(np.count_nonzero(grid["data"] == -1))},
                       "occupancy_image_matches_grid": True,
                       "occupancy_image_size": [150, 200],
                       "occupancy_image_orientation": "forward_up_left_to_left"}
            if scene["kind"] in ("flat", "slope"):
                truth = scene["truth"]
                ground_accuracy = float(np.mean(predicted[truth == 2] == 2))
                obstacle_accuracy = float(np.mean(predicted[truth == 3] == 3))
                if ground_accuracy < 0.95 or obstacle_accuracy < 0.95:
                    raise ValueError(f"{stage}: known plane/box labels disagree ({ground_accuracy:.3f}/{obstacle_accuracy:.3f})")
                if not np.all(predicted[truth == 1] == 1):
                    raise ValueError(f"{stage}: unsupported far return was classified as known")
                def at(x, y):
                    column = int(math.floor((x - grid["origin"][0]) / grid["resolution"]))
                    row = int(math.floor((y - grid["origin"][1]) / grid["resolution"]))
                    return int(grid["data"][row, column])
                # Coordinates address distinct x/y cells, detecting a transposed grid.
                probes = {"box_front_occupied": at(3.1, 0.1), "corner_occupied": at(5.5, 2.1),
                          "direct_ground_free": at(1.1, 0.1), "box_shadow_unknown": at(5.1, 0.1),
                          "unobserved_far_unknown": at(25.1, 9.1)}
                required = {"box_front_occupied": 100, "corner_occupied": 100,
                            "direct_ground_free": 0, "box_shadow_unknown": -1, "unobserved_far_unknown": -1}
                if probes != required:
                    raise ValueError(f"{stage}: occupied/free/unknown occlusion or row-major placement differs: {probes}")
                outcome.update(synthetic_ground_label_fraction=ground_accuracy,
                               synthetic_obstacle_label_fraction=obstacle_accuracy, cell_probes=probes)
            elif scene["kind"] == "no_floor":
                if counts["ground"] or counts["obstacle"] or np.any(grid["data"] != -1):
                    raise ValueError("No-floor scene invented ground, obstacles, or known occupancy")
                if not np.isnan(record["unknown"]["height"]).all():
                    raise ValueError("No-floor returns were assigned an invented height_above_ground")
                outcome["no_floor_rejected_without_guessing"] = True
            elif scene["kind"] == "vehicle_mask":
                truth = scene["truth"]
                start = scene["explicit_edge_start"]
                if not np.array_equal(predicted[start:], np.where(truth[start:] == 4, 0, truth[start:])):
                    raise ValueError("Body mask lost or mislabeled a known point just outside a measured edge")
                if np.mean(predicted[truth == 2] == 2) < 0.95 or np.mean(predicted[truth == 3] == 3) < 0.95:
                    raise ValueError("Vehicle mask distorted surrounding known floor or external obstacle labels")
                if len(output_indices) + int(np.count_nonzero(masked)) + int(np.count_nonzero(~valid)) != len(source_xyz):
                    raise ValueError("Masked, invalid, and emitted partitions do not account for every input point")
                outcome.update(body_mask_bounds=VEHICLE_MASK, explicit_edge_points=len(truth)-start,
                               near_outside_obstacles_preserved=True,
                               masks_vehicle_body_only_mirrors_unmodeled=True)
            report["stage_results"][stage] = outcome
            report["verified_by_stage"][stage] = report["verified_by_stage"].get(stage, 0) + 1
            verified.add(key)

        def receive_cloud(name, message):
            try:
                report["output_messages"] += 1
                if message.header.frame_id != "os_sensor":
                    raise ValueError("Classified cloud header is not os_sensor")
                key = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                metadata, payload = cloud_metadata(message), bytes(message.data)
                if not message.width * message.height:
                    if payload:
                        raise ValueError("Empty classified cloud contains payload")
                    report["clear_cloud_messages"] += 1
                    output = {"indices": np.empty(0, dtype=np.uint32), "xyz": np.empty((0, 3)),
                              "height": np.empty(0), "labels": np.empty(0, dtype=np.uint8)}
                else:
                    if key not in published or published[key]["stage"].startswith("wrong"):
                        raise ValueError("Nonempty classified output has an unknown or wrong-frame source header")
                    decoded = decode_point_cloud(metadata, payload)
                    indices = scalar_field(metadata, payload, "source_index", 6)
                    labels = scalar_field(metadata, payload, "classification", 2)
                    rgb = scalar_field(metadata, payload, "rgb") & 0x00ffffff
                    heights = scalar_field(metadata, payload, "height_above_ground", 7)
                    if name == "classified":
                        if not np.isin(labels, [1, 2, 3]).all():
                            raise ValueError("Combined cloud contains an invalid or masked class")
                        for classification, color in label_semantics.values():
                            if not np.all(rgb[labels == classification] == color):
                                raise ValueError("Combined cloud diagnostic RGB differs from its classification")
                    else:
                        classification, color = label_semantics[name]
                        if not np.all(labels == classification) or not np.all(rgb == color):
                            raise ValueError(f"{name} output has inconsistent class or diagnostic RGB")
                    if not np.isfinite(heights[labels != 1]).all():
                        raise ValueError("Known point heights must be finite")
                    output = {"indices": indices, "xyz": decoded.xyz, "height": heights, "labels": labels}
                last_output[name] = len(output["indices"])
                if key in published:
                    record = output_records.setdefault(key, {})
                    if name in record:
                        raise ValueError("A consumed source cloud was republished")
                    record[name] = output
                    check_result(key)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                record_error(error)

        def receive_grid(message):
            try:
                report["output_messages"] += 1
                if message.header.frame_id != "os_sensor":
                    raise ValueError("Occupancy header is not os_sensor")
                info = message.info
                q = info.origin.orientation
                if (info.width != 200 or info.height != 150 or abs(info.resolution - 0.2) > 1e-6
                        or not np.allclose([info.origin.position.x, info.origin.position.y, info.origin.position.z],
                                           [0, -15, 0], atol=1e-6)
                        or not np.allclose([q.x, q.y, q.z, q.w], [0, 0, 0, 1], atol=1e-6)):
                    raise ValueError("Occupancy dimensions/origin do not describe 40 x 30 m at 20 cm in os_sensor")
                data = np.asarray(message.data, dtype=np.int16)
                if data.size != info.width * info.height or not np.isin(data, [-1, 0, 100]).all():
                    raise ValueError("Occupancy values must be exactly unknown=-1, free=0, occupied=100")
                data = data.reshape(info.height, info.width)
                last_output["grid_known"] = int(np.count_nonzero(data != -1))
                if not last_output["grid_known"]:
                    report["clear_grid_messages"] += 1
                key = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                if key not in published:
                    if last_output["grid_known"]:
                        raise ValueError("Nonempty occupancy has an unknown source header")
                    if active_stage == "stale":
                        stale_grid_stamps.add(key)
                    return
                record = output_records.setdefault(key, {})
                if "grid" in record:
                    raise ValueError("A consumed occupancy source cloud was republished")
                record["grid"] = {"data": data, "origin": [info.origin.position.x, info.origin.position.y],
                                  "resolution": float(info.resolution)}
                check_result(key)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                record_error(error)

        def receive_occupancy_image(message):
            try:
                report["output_messages"] += 1
                if message.header.frame_id != "os_sensor":
                    raise ValueError("Occupancy image header is not os_sensor")
                if (message.width != 150 or message.height != 200 or message.encoding != "rgb8"
                        or message.is_bigendian or message.step < 150*3):
                    raise ValueError("Occupancy image must be native 150 x 200 RGB8 with a valid row stride")
                payload = bytes(message.data)
                if len(payload) != message.step * message.height:
                    raise ValueError("Occupancy image data disagrees with its layout")
                pixels = np.ndarray((message.height, message.width, 3), dtype=np.uint8, buffer=payload,
                                    strides=(message.step, 3, 1)).copy()
                last_output["image_known"] = int(np.count_nonzero(np.any(pixels != 48, axis=2)))
                if not last_output["image_known"]:
                    report["clear_image_messages"] += 1
                key = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                if key not in published:
                    if last_output["image_known"]:
                        raise ValueError("Known occupancy image has an unknown source header")
                    if active_stage == "stale":
                        stale_image_stamps.add(key)
                    return
                if published[key]["stage"].startswith("wrong") and last_output["image_known"]:
                    raise ValueError("Wrong-frame input produced known occupancy image pixels")
                record = output_records.setdefault(key, {})
                if "image" in record:
                    raise ValueError("A consumed occupancy image source cloud was republished")
                record["image"] = pixels
                check_result(key)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                record_error(error)

        def receive_diagnostics(message):
            try:
                for status in message.status:
                    if status.name.split("/")[-1] != "ground_bev":
                        continue
                    values = {item.key: item.value for item in status.values}
                    level = status.level[0] if isinstance(status.level, (bytes, bytearray)) else int(status.level)
                    snapshot = {"level": level, "message": status.message, "values": values}
                    report["last_diagnostic"] = snapshot
                    diagnostic_by_stage.setdefault(active_stage, []).append(snapshot)
                    if values.get("provisional") != "true" or values.get("stationary_mode") != "true":
                        raise ValueError("Geometry diagnostics omitted provisional stationary qualification")
                    report["provisional_diagnostics"] += 1
                    if active_stage == "vehicle_mask" and int(values.get("input_points", "0")) == len(scenes["vehicle_mask"]["xyz"]):
                        truth = scenes["vehicle_mask"]["truth"]
                        if (int(values.get("masked_points", "-1")) != int(np.count_nonzero(truth == 4))
                                or int(values.get("invalid_points", "-1")) != int(np.count_nonzero(truth == 0))):
                            raise ValueError("Vehicle mask diagnostics do not account for the known body/invalid returns")
                        for name, value in VEHICLE_MASK.items():
                            if isinstance(value, bool):
                                correct = values.get(name) == str(value).lower()
                            else:
                                correct = abs(float(values.get(name, "nan")) - value) < 1e-6
                            if not correct:
                                raise ValueError(f"Vehicle body bound {name} differs from supplied measurements")
                        report["mask_diagnostics_verified"] += 1
            except (ValueError, TypeError, KeyError) as error:
                record_error(error)

        subscriptions = [node.create_subscription(PointCloud2, topics[name + "_topic"],
                                                   lambda message, name=name: receive_cloud(name, message), reliable)
                         for name in (*label_semantics, "classified")]
        subscriptions += [node.create_subscription(OccupancyGrid, topics["occupancy_topic"], receive_grid, reliable),
                          node.create_subscription(Image, topics["occupancy_image_topic"], receive_occupancy_image, reliable),
                          node.create_subscription(DiagnosticArray, topics["diagnostics_topic"], receive_diagnostics, 5)]
        # Refuse other nodes in the supposedly isolated domain before starting ours.
        discovery_until = time.monotonic() + 1.0
        while time.monotonic() < discovery_until:
            rclpy.spin_once(node, timeout_sec=0.05)
        others = [(name, namespace) for name, namespace in node.get_node_names_and_namespaces()
                  if name != "ground_bev_isolated_check"]
        if others:
            raise ValueError(f"Private ROS domain is occupied: {others}")
        arguments = [str(args.executable), "--ros-args", "-r", "__node:=ground_bev_check_cpp"]
        parameters = dict(topics, expected_frame="os_sensor", stale_after_s=0.45, max_publish_hz=20.0,
                          ground_height_m=1.8, ground_height_tolerance_m=0.35, self_mask_enabled=False)
        for name, value in parameters.items():
            value = str(value).lower() if isinstance(value, bool) else value
            arguments += ["-p", f"{name}:={value}"]
        report["node_arguments"] = arguments
        with child_process(arguments, report) as process:
            def spin():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Bounded ground BEV verifier deadline exceeded")
                if process.poll() is not None:
                    raise RuntimeError(f"C++ ground node exited with {process.returncode}")
                rclpy.spin_once(node, timeout_sec=0.02)
                if report["errors"]:
                    raise ValueError(report["errors"][0])

            discovery_until = min(deadline, time.monotonic() + 5.0)
            while not publisher.get_subscription_count():
                spin()
                if time.monotonic() >= discovery_until:
                    raise TimeoutError("C++ node did not subscribe to private input")

            def publish(stage, scene, wrong_frame=False):
                message = PointCloud2()
                stamp = node.get_clock().now().nanoseconds
                message.header.frame_id = "os_lidar" if wrong_frame else "os_sensor"
                message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
                for name in ("width", "height", "point_step", "row_step", "is_bigendian", "is_dense"):
                    setattr(message, name, scene["metadata"][name])
                message.fields = [PointField(**item) for item in scene["metadata"]["fields"]]
                message.data = scene["data"]
                published[stamp] = {"stage": stage, "scene": scene}
                report["published_input_messages"] += 1
                publisher.publish(message)

            def drive(stage, scene, seconds, required=2, wrong_frame=False):
                nonlocal active_stage
                active_stage = stage
                until = time.monotonic() + seconds
                next_publish = 0.0
                while time.monotonic() < until or report["verified_by_stage"].get(stage, 0) < required:
                    if time.monotonic() >= next_publish:
                        publish(stage, scene, wrong_frame)
                        next_publish = time.monotonic() + 0.12
                    spin()

            drive("wrong_frame_initial", scenes["flat"], 1.25, required=0, wrong_frame=True)
            if any(last_output.get(name, 0) for name in (*label_semantics, "classified", "grid_known", "image_known")):
                raise ValueError("Initial wrong-frame input did not leave empty clouds and unknown grid")
            drive("flat", scenes["flat"], 1.2)
            drive("slope", scenes["slope"], 1.2)
            drive("no_floor", scenes["no_floor"], 1.2)
            if "fixture" in scenes:
                drive("fixture", scenes["fixture"], 1.4)
            drive("wrong_frame_after_valid", scenes["flat"], 1.25, required=0, wrong_frame=True)
            if any(last_output.get(name, 0) for name in (*label_semantics, "classified", "grid_known", "image_known")):
                raise ValueError("Wrong-frame input did not clear previous geometry")
            drive("recovered", scenes["flat"], 1.0)
            active_stage = "stale"
            stopped = time.monotonic()
            while time.monotonic() - stopped < 1.3:
                spin()
            if any(last_output.get(name, 0) for name in (*label_semantics, "classified", "grid_known", "image_known")):
                raise ValueError("Stale geometry did not clear to empty clouds and unknown occupancy")
            if not all(report[name] for name in ("clear_cloud_messages", "clear_grid_messages", "clear_image_messages")):
                raise ValueError("No explicit stale/wrong-input clearing was observed")
            if not stale_grid_stamps.intersection(stale_image_stamps):
                raise ValueError("Stale occupancy image and grid did not clear with the same new header")
            if not report["provisional_diagnostics"]:
                raise ValueError("No provisional stationary geometry diagnostics were received")
            for stage in ("wrong_frame_initial", "wrong_frame_after_valid"):
                if not any("frame" in item["message"].lower() or "frame" in item["values"].get("current_error", "").lower()
                           for item in diagnostic_by_stage.get(stage, [])):
                    raise ValueError(f"{stage}: wrong frame was not explained in diagnostics")
            report["wrong_frame_rejected_before_and_after_valid_input"] = True
            report["stale_output_cleared"] = True
            report["stale_occupancy_image_cleared"] = True
            report["old_source_republication_observed"] = False
        if args.check_vehicle_mask:
            active_stage = "mask_discovery"
            mask_arguments = [str(args.executable), "--ros-args", "-r", "__node:=ground_bev_mask_check_cpp"]
            mask_parameters = dict(parameters, **VEHICLE_MASK)
            for name, value in mask_parameters.items():
                value = str(value).lower() if isinstance(value, bool) else value
                mask_arguments += ["-p", f"{name}:={value}"]
            report["mask_node"] = {"node_arguments": mask_arguments}
            with child_process(mask_arguments, report["mask_node"]) as process:
                discovery_until = min(deadline, time.monotonic() + 4.0)
                while not publisher.get_subscription_count():
                    spin()
                    if time.monotonic() >= discovery_until:
                        raise TimeoutError("Configured mask child did not subscribe to private input")
                # A second process is necessary: mask parameters are fixed at
                # construction, so the ordinary plane/slope checks stay intact.
                drive("vehicle_mask", scenes["vehicle_mask"], 1.4)
                while not report["mask_diagnostics_verified"]:
                    publish("vehicle_mask", scenes["vehicle_mask"])
                    spin()
                report["vehicle_mask_software_check_passed"] = True
        report["diagnostics_by_stage"] = {stage: items[-1] for stage, items in diagnostic_by_stage.items() if items}
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
    except Exception as error:
        if str(error) not in report["errors"]:
            report["errors"].append(str(error))
    finally:
        if node is not None:
            node.destroy_node()
        if "rclpy" in locals() and rclpy.ok():
            rclpy.shutdown()
        report["elapsed_s"] = round(time.monotonic() - started, 6)
        report["checks_passed"] = not report["errors"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"checks_passed": report["checks_passed"], "verified_by_stage": report["verified_by_stage"],
                          "errors": report["errors"], "evidence_file": str(args.output)}, indent=2))
    return 0 if report["checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
