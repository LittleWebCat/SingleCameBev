#!/usr/bin/env python3
"""Summarize a scripts/record_all_cameras_perf.sh run directory (stdout, plain text)."""
import csv
import re
import statistics
import sys
from pathlib import Path


def stats(values):
    v = sorted(values)
    if not v:
        return 'no samples'
    return (f'median {statistics.median(v):.1f}  mean {statistics.fmean(v):.1f}  '
            f'p95 {v[min(len(v) - 1, int(0.95 * len(v)))]:.1f}  max {v[-1]:.1f}  (n={len(v)})')


def floats(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def main():
    run = Path(sys.argv[1])
    print(f'Performance summary: {run.name}')
    info = run / 'run_info.txt'
    if info.exists():
        print(info.read_text().rstrip())
    print()

    lidar = run / 'lidar_detector.csv'
    if lidar.exists():
        rows = list(csv.DictReader(lidar.open()))
        rows = [r for r in rows if r.get('detection_hz') not in ('', '0.00')]
        print(f'LiDAR detector ({rows[-1]["model"] if rows else "no data"}), per-second samples:')
        for key, unit in [('scan_hz', 'Hz'), ('detection_hz', 'Hz'), ('latency_median_ms', 'ms'),
                          ('processing_ms', 'ms'), ('detector_ms', 'ms'), ('preprocess_ms', 'ms'),
                          ('inference_ms', 'ms'), ('postprocess_ms', 'ms'), ('model_input_points', 'pts'),
                          ('object_boxes', 'boxes')]:
            print(f'  {key:20s} {stats(floats(rows, key))} {unit}')
        print()

    multicam = run / 'multicam.log'
    if multicam.exists():
        text = multicam.read_text(errors='replace')
        fps = [float(x) for x in re.findall(r'processing_fps=([\d.]+)', text)]
        print('Multi-camera pipeline (6 cameras):')
        print(f'  processing_fps (running average) last {fps[-1]:.1f}' if fps else '  no status lines')
        for line in text.splitlines():
            if line.startswith('Finished:') or 'approx_fps=' in line:
                print('  ' + line.strip())
        errors = [l for l in text.splitlines() if re.search(r'error|NO FRAMES|failed', l, re.I)]
        if errors:
            print(f'  {len(errors)} error lines, first: {errors[0].strip()[:160]}')
        print()

    gpu = run / 'gpu_load.csv'
    if gpu.exists():
        util, power = [], []
        for row in csv.reader(gpu.open()):
            try:
                util.append(float(row[1].strip().rstrip(' %')))
                power.append(float(row[3].strip().split()[0]))
            except (IndexError, ValueError):
                pass
        print(f'GPU utilization %   {stats(util)}')
        if power:
            print(f'GPU power W         {stats(power)}')

    tegra = run / 'tegrastats.log'
    if tegra.exists():
        ram, cpu, temp, vin = [], [], [], []
        for line in tegra.read_text().splitlines():
            if m := re.search(r'RAM (\d+)/', line):
                ram.append(float(m.group(1)) / 1024)
            if m := re.search(r'CPU \[([^\]]+)\]', line):
                cores = [float(c.split('%')[0]) for c in m.group(1).split(',') if '%' in c]
                if cores:
                    cpu.append(statistics.fmean(cores))
            temps = [float(t) for t in re.findall(r'@([\d.]+)C', line)]
            if temps:
                temp.append(max(temps))
            if m := re.search(r'VIN (\d+)mW', line):
                vin.append(float(m.group(1)) / 1000)
        print(f'CPU average %       {stats(cpu)}')
        print(f'RAM used GiB        {stats(ram)}')
        print(f'Hottest sensor C    {stats(temp)}')
        print(f'Board input W       {stats(vin)}')

    video = run / 'screen.mp4'
    print()
    print(f'Screen video: {video} ({video.stat().st_size / 1e6:.0f} MB)' if video.exists() else 'Screen video: missing')


if __name__ == '__main__':
    main()
