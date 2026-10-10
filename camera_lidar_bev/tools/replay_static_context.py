#!/usr/bin/python3
"""Restore the exact saved Ouster static context before a replay start offset.

Only /tf_static and /metadata are read. Their last recorded messages before the
selected offset are republished unchanged, with reliable transient-local QoS.
The publisher stays alive so later subscribers can receive the saved context.
No sensor, transform or calibration is created or configured by this helper.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import yaml


TOPICS = ('/tf_static', '/metadata')


def positive_offset(value):
    offset = float(value)
    if not math.isfinite(offset) or offset <= 0:
        raise argparse.ArgumentTypeError('start offset must be finite and greater than zero')
    return offset


def load_context(bag, offset):
    """Use indexed topic filtering; never scan image or point-cloud payloads."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    bag = Path(bag).expanduser().resolve()
    metadata_file = bag/'metadata.yaml'
    if not metadata_file.is_file():
        raise ValueError(f'recording metadata is missing: {metadata_file}')
    if not math.isfinite(offset) or offset <= 0:
        raise ValueError('start offset must be finite and greater than zero')
    metadata = yaml.safe_load(metadata_file.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f'invalid bag metadata: {metadata_file}')
    try:
        info = metadata['rosbag2_bagfile_information']
        start = int(info['starting_time']['nanoseconds_since_epoch'])
        duration = int(info['duration']['nanoseconds'])
        storage = info['storage_identifier']
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'invalid bag timing/storage metadata: {metadata_file}') from error
    if start <= 0 or duration <= 0 or not isinstance(storage, str) or not storage:
        raise ValueError(f'invalid bag timing/storage metadata: {metadata_file}')
    if offset*1e9 >= duration:
        raise ValueError(f'start offset {offset:g}s is outside the {duration/1e9:g}s recording')
    cutoff = start+int(offset*1e9)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_read_order(rosbag2_py.ReadOrder(rosbag2_py.ReadOrderSortBy.ReceivedTimestamp))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(TOPICS)))
    saved = {}
    while reader.has_next():
        topic, serialized, recorded_stamp = reader.read_next()
        if recorded_stamp >= cutoff:
            break
        saved[topic] = (serialized, recorded_stamp)
    missing = [topic for topic in TOPICS if topic not in saved]
    if missing:
        raise ValueError(f'no saved {", ".join(missing)} before replay offset {offset:g}s')
    types = {'/tf_static': TFMessage, '/metadata': String}
    messages = {topic: deserialize_message(saved[topic][0], types[topic]) for topic in TOPICS}
    static = messages['/tf_static']
    if not static.transforms or any(not item.header.frame_id or not item.child_frame_id
                                    for item in static.transforms):
        raise ValueError('saved /tf_static has no usable named transforms')
    try:
        sensor_metadata = json.loads(messages['/metadata'].data)
    except (TypeError, ValueError) as error:
        raise ValueError('saved /metadata is not valid JSON') from error
    if not isinstance(sensor_metadata, dict) or not sensor_metadata:
        raise ValueError('saved /metadata must be a nonempty JSON object')
    return messages, {topic: (saved[topic][1]-start)/1e9 for topic in TOPICS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True, type=Path, help='recorded bag directory')
    parser.add_argument('--start-offset', required=True, type=positive_offset,
                        help='replay start in recorded seconds; must be inside the bag')
    args = parser.parse_args(argv)
    try:
        messages, offsets = load_context(args.bag, args.start_offset)
    except Exception as error:
        print(f'Replay static context failed: {error}', file=sys.stderr, flush=True)
        return 1

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    node = None
    try:
        rclpy.init()
        node = Node('recorded_static_context', enable_rosout=False)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        publishers = {
            '/tf_static': node.create_publisher(TFMessage, '/tf_static', qos),
            '/metadata': node.create_publisher(String, '/metadata', qos),
        }
        for topic in TOPICS:
            publishers[topic].publish(messages[topic])
        print('Restored exact recorded static context: '+', '.join(
            f'{topic} from {offsets[topic]:.6f}s' for topic in TOPICS)+
            '. Original TF stamps and metadata are preserved.', flush=True)
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception as error:
        print(f'Replay static context publisher failed: {error}', file=sys.stderr, flush=True)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
