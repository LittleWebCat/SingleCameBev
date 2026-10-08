#!/usr/bin/env python3
"""Rebuild the pinned PointPillars ONNX with the installed native scatter plugin."""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True,
                        help='new model directory; existing directories are refused')
    parser.add_argument('--plugin-library', type=Path,
                        default=root / 'install/bev_lidar_detection/lib/libpointpillars_core.so')
    parser.add_argument('--trtexec', type=Path, default=Path('/usr/src/tensorrt/bin/trtexec'))
    args = parser.parse_args()
    source = root / 'models/lidar_objects_001'
    provenance = json.loads((source / 'provenance.json').read_text())
    onnx = source / provenance['onnx_file']
    if sha256(onnx) != provenance['onnx_sha256']:
        parser.error('Pinned source ONNX hash mismatch')
    for required in (args.plugin_library, args.trtexec):
        if not required.is_file():
            parser.error(f'Missing required runtime/build artifact: {required}')
    out = args.output_directory.resolve()
    if out.exists():
        parser.error(f'Refusing to overwrite existing directory: {out}')
    out.mkdir(parents=True)
    shutil.copy2(onnx, out / 'pointpillar.onnx')
    shutil.copy2(source / provenance['reference_scan_file'], out / provenance['reference_scan_file'])
    command = [str(args.trtexec), f'--onnx={out / "pointpillar.onnx"}', '--fp16',
               '--inputIOFormats=fp16:chw,int32:chw,int32:chw',
               f'--staticPlugins={args.plugin_library.resolve()}',
               f'--saveEngine={out / "pointpillar_fp16.plan"}',
               '--skipInference', '--memPoolSize=workspace:2048']
    provenance.update(engine_sha256=None, engine_build_complete=False,
                      engine_build_command=command,
                      plugin_library_sha256=sha256(args.plugin_library))
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    start = time.monotonic()
    log = out / 'build_engine.log'
    with log.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f'Engine build failed ({result.returncode}); inspect {log}')
    engine = out / 'pointpillar_fp16.plan'
    provenance.update(engine_file=engine.name, engine_sha256=sha256(engine),
                      engine_build_complete=True, engine_bytes=engine.stat().st_size,
                      engine_build_seconds=time.monotonic() - start, engine_build_log=log.name,
                      ouster_accuracy_validated=False, metric_size_accuracy_validated=False)
    for key in ('reference_benchmark', 'ouster_benchmark', 'cuda_memcheck'):
        provenance.pop(key, None)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Built {engine}; SHA256 {provenance["engine_sha256"]}')


if __name__ == '__main__':
    main()
