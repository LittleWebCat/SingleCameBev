#!/usr/bin/env python3
"""Render camera road/lane masks beside LiDAR BEV without road geometry."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from road_lane_detector import TwinLiteRuntime
from combined_bev_display import CombinedBevRenderer


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-id', default='002')
    parser.add_argument('--manifest', type=Path, default=root/'data/perception_compare_001/manifest.json')
    parser.add_argument('--results', type=Path, default=root/'evaluation/perception_speed_compare_001.json')
    parser.add_argument('--engine', type=Path, default=root/'models/road_lanes_001/twinlitenet_fp16.engine')
    parser.add_argument('--output-dir', type=Path, default=root/'data/road_lane_preview_001')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a new output directory')
    manifest = json.loads(args.manifest.read_text())
    results = json.loads(args.results.read_text())
    source = next((s for s in manifest['samples'] if s['sample_id'] == args.sample_id), None)
    sample = next((s for s in results['samples'] if s['sample_id'] == args.sample_id), None)
    if source is None or sample is None:
        parser.error('Sample ID is missing from the cached comparison')
    image = cv2.imread(source['image'])
    if image is None:
        parser.error('Cannot read cached camera image')
    points = np.fromfile(source['cloud'], np.float32).reshape(-1, 4)
    model = TwinLiteRuntime(args.engine)
    masks, runtime_ms, engine_ms = model.infer(image)
    calibration = manifest['calibration']
    renderer = CombinedBevRenderer(calibration)
    canvas = renderer.render(image, sample, points,
                             status={'mode': 'saved camera + LiDAR sample', 'TwinLiteNet': f'{runtime_ms:.1f} ms single frame',
                                     'alignment': 'saved candidate B'},
                             segmentation={'road_mask': masks[0], 'lane_mask': masks[1]})
    args.output_dir.mkdir(parents=True)
    for name, output in [('combined.png', canvas), ('road_mask.png', masks[0]), ('lane_mask.png', masks[1])]:
        if not cv2.imwrite(str(args.output_dir/name), output):
            raise RuntimeError('Cannot save '+name)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'sample_id': args.sample_id,
              'source_image': source['image'], 'source_cloud': source['cloud'],
              'model': 'TwinLiteNet', 'backend': 'TensorRT FP16',
              'single_frame_runtime_ms': runtime_ms, 'single_frame_engine_ms': engine_ms,
              'camera_road_pixels': int(np.count_nonzero(masks[0])), 'camera_lane_pixels': int(np.count_nonzero(masks[1])),
              'road_geometry_enabled': False, 'segmentation_coordinates': 'native camera-image pixels',
              'physical_accuracy_validated': False,
              'artifacts': {name: str((args.output_dir/name).resolve()) for name in ('combined.png', 'road_mask.png', 'lane_mask.png')}}
    (args.output_dir/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
