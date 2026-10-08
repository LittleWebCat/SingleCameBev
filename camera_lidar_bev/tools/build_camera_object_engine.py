#!/usr/bin/env python3
"""Build a traceable batch-one camera detector from existing local YOLO11 weights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=project.parent / 'yolo11n.pt')
    parser.add_argument('--output-directory', type=Path, required=True,
                        help='new directory; existing exports are never replaced')
    args = parser.parse_args()
    source = args.weights.resolve()
    destination = args.output_directory.resolve()
    if not source.is_file():
        parser.error('weights must be an existing local file; this builder does not download models')
    if destination.exists() or destination.is_symlink():
        parser.error('output-directory already exists; choose a new model version')
    destination.mkdir(parents=True, exist_ok=False)
    settings = destination / 'ultralytics_settings'
    (settings / 'Ultralytics').mkdir(parents=True)
    os.environ['YOLO_CONFIG_DIR'] = str(settings)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    weights = destination / 'yolo11n.pt'
    shutil.copyfile(source, weights)

    import ultralytics
    from ultralytics import YOLO
    import tensorrt as trt

    model = YOLO(str(weights), task='detect')
    names = {int(key): str(value) for key, value in model.names.items()}
    expected = {0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}
    if len(names) != 80 or any(names.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Weights do not have the expected COCO road-user classes')
    engine_path = Path(model.export(format='engine', imgsz=640, batch=1, dynamic=False,
                                   simplify=False, quantize=16, device=0, workspace=1, opset=17))
    data = engine_path.read_bytes()
    size = int.from_bytes(data[:4], 'little')
    if not 0 < size < min(len(data)-4, 1024*1024):
        raise RuntimeError('Export did not have the expected Ultralytics metadata prefix')
    metadata = json.loads(data[4:4+size])
    if {int(key): str(value) for key, value in metadata['names'].items()} != names:
        raise RuntimeError('Exported model class names differ from the source weights')
    raw = data[4+size:]
    raw_path = destination / 'yolo11n_fp16_raw.engine'
    raw_path.write_bytes(raw)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(raw)
    if engine is None:
        raise RuntimeError('Fresh engine could not be deserialized on this machine')
    bindings = [{'name':engine.get_tensor_name(index),
                 'mode':str(engine.get_tensor_mode(engine.get_tensor_name(index))),
                 'shape':list(engine.get_tensor_shape(engine.get_tensor_name(index))),
                 'dtype':str(engine.get_tensor_dtype(engine.get_tensor_name(index)))}
                for index in range(engine.num_io_tensors)]
    report = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'source_weights':str(source),
              'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
              'raw_engine_sha256':hashlib.sha256(raw).hexdigest(),
              'raw_engine_file':str(raw_path), 'exported_engine_file':str(engine_path),
              'ultralytics_version':ultralytics.__version__, 'tensorrt_version':trt.__version__,
              'export':{'task':'detect','imgsz':640,'batch':1,'dynamic':False,'quantize':16,'opset':17},
              'classes':names, 'bindings':bindings, 'export_metadata':metadata,
              'model_license_from_export':metadata.get('license'),
              'accuracy_on_live_camera_validated':False, 'speed_on_live_camera_measured':False}
    provenance = destination / 'provenance.json'
    provenance.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'raw_engine_file':str(raw_path),'bindings':bindings,
                      'provenance':str(provenance)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
