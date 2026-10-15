#!/usr/bin/python3
"""Inspect raw return support near camera vehicle proposals, without inference.

Image ROIs are proposals, not verified point membership. Depth bands and a
nearby image patch expose foreground/background groups and projection offsets.
This tool neither fits road geometry nor changes camera/LiDAR calibration.
"""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from combined_bev_display import calibration_arrays, project_camera


def case_argument(value):
    fields = value.split(':')
    if len(fields) != 4:
        raise argparse.ArgumentTypeError('Use sample_id:box_id:min_range:max_range')
    sid, index, lower, upper = fields
    try:
        result = (sid, int(index), float(lower), float(upper))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if result[1] < 0 or not 0 <= result[2] < result[3]:
        raise argparse.ArgumentTypeError('Invalid box index or range band')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case', action='append', type=case_argument, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a new output directory')
    manifest = json.loads((args.samples/'manifest.json').read_text())
    detections = json.loads((args.samples/'detections.json').read_text())
    sources = {s['sample_id']: s for s in manifest['samples']}
    labels = {s['sample_id']: s['detections'] for s in detections['samples']}
    calibration = manifest['calibration']
    transform = calibration_arrays(calibration)[2]
    args.output_dir.mkdir(parents=True)
    plt.style.use('dark_background')
    report_cases, interactive = [], []
    for sid, box_id, lower, upper in args.case:
        source = sources[sid]
        box = labels[sid][box_id]
        image_path = (args.samples/source['image']).resolve()
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise RuntimeError(f'Cannot read {image_path}')
        xyz = np.fromfile(args.samples/source['cloud'], np.float32).reshape(-1, 4)[:, :3]
        ranges = np.linalg.norm(xyz, axis=1)
        pixels, valid = project_camera(xyz@transform[:3, :3].T+transform[:3, 3], calibration)
        roi = np.asarray(box['native_xyxy_px'])
        patch = np.array([max(0, roi[0]-10), max(0, roi[1]-10),
                          min(bgr.shape[1]-1, roi[2]+10), min(bgr.shape[0]-1, roi[3]+100)])
        def inside(bounds):
            return (valid & (pixels[:, 0] >= bounds[0]) & (pixels[:, 0] <= bounds[2]) &
                    (pixels[:, 1] >= bounds[1]) & (pixels[:, 1] <= bounds[3]) & (ranges >= .5))
        in_roi, in_patch = inside(roi), inside(patch)
        selected = in_patch & (ranges >= lower) & (ranges <= upper)
        selected_roi = in_roi & selected
        points = xyz[selected]
        count, core_count = int(selected.sum()), int(selected_roi.sum())
        measured = dict(sample_id=sid, box_id=box_id, elapsed_s=source['elapsed_s'],
                        camera_class=box['class_name'], camera_score=box['score'],
                        image_stamp_ns=source['image_stamp_ns'], lidar_stamp_ns=source['lidar_stamp_ns'],
                        header_delta_ms=source['pair_delta_ms'], camera_roi_xyxy=roi.tolist(),
                        search_patch_xyxy=patch.tolist(), selected_range_band_m=[lower, upper],
                        raw_returns_in_camera_roi=int(in_roi.sum()),
                        selected_raw_returns_in_camera_roi=core_count,
                        selected_raw_returns_in_search_patch=count,
                        selected_median_range_m=float(np.median(ranges[selected])) if count else None,
                        selected_visible_extent_xyz_m=np.ptp(points, axis=0).tolist() if count else None,
                        counts_are_verified_vehicle_membership=False,
                        vehicle_range_is_ground_truth=False)
        key = f'{sid}_box{box_id}_{lower:g}_{upper:g}m'
        annotated = bgr.copy()
        for point in pixels[in_patch]:
            cv2.circle(annotated, tuple(np.rint(point).astype(int)), 1, (130, 130, 130), -1)
        for point in pixels[selected]:
            cv2.circle(annotated, tuple(np.rint(point).astype(int)), 2, (35, 170, 255), -1)
        cv2.rectangle(annotated, tuple(np.rint(roi[:2]).astype(int)),
                      tuple(np.rint(roi[2:]).astype(int)), (30, 245, 245), 1)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=130,
                                 gridspec_kw={'height_ratios': [1.1, 1]})
        fig.suptitle(f'10 Hz raw LiDAR | frame {sid}, t={source["elapsed_s"]:.1f}s | '
                     f'{box["class_name"]} proposal | {lower:g}–{upper:g} m return band', fontsize=16)
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('Camera view: yellow vehicle proposal; orange selected raw returns')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(rgb)
        axes[0, 1].set_xlim(patch[0]-15, patch[2]+15)
        axes[0, 1].set_ylim(patch[3]+15, patch[1]-15)
        axes[0, 1].add_patch(Rectangle(patch[:2], patch[2]-patch[0], patch[3]-patch[1],
                                     fill=False, edgecolor='#afb8c3', linestyle='--'))
        axes[0, 1].set_title(f'Zoom: {core_count} selected returns inside box; {count} in nearby patch')
        axes[0, 1].set_xlabel('Native image u (px)')
        axes[0, 1].set_ylabel('Native image v (px)')
        if count:
            axes[1, 0].scatter(points[:, 1], points[:, 2], c=ranges[selected],
                               cmap='autumn', s=18, vmin=lower, vmax=upper)
            axes[1, 0].set_aspect('equal', adjustable='datalim')
        else:
            axes[1, 0].text(.5, .5, 'No selected raw returns', ha='center', va='center',
                            transform=axes[1, 0].transAxes)
        axes[1, 0].set_xlabel('Sensor Y: left (m)')
        axes[1, 0].set_ylabel('Sensor Z: up (m)')
        axes[1, 0].set_title('Selected raw point profile; visible surfaces, no predicted box')
        axes[1, 0].grid(alpha=.2)
        maximum = max(100., upper+15, float(np.percentile(ranges[in_patch], 99)) if in_patch.any() else 100.)
        bins = np.arange(0, min(260., maximum)+1, 1.)
        axes[1, 1].hist(ranges[in_patch], bins=bins, color='#8394a7', label='Nearby image patch')
        axes[1, 1].hist(ranges[in_roi], bins=bins, color='#f7ec58', alpha=.75, label='Exact vehicle ROI')
        axes[1, 1].axvspan(lower, upper, color='#ff9b30', alpha=.3, label='Selected depth band')
        axes[1, 1].set_xlabel('Slant range from os_sensor origin (m)')
        axes[1, 1].set_ylabel('Raw returns per 1 m bin')
        axes[1, 1].set_title('Depth separation exposes background and overlapping surfaces')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=.2)
        fig.text(.02, .018, 'Single raw scan, all selected returns counted. ROI/band points can include road and background. '
                 'Candidate B calibration and physical sensor timing remain unverified.', fontsize=9)
        fig.tight_layout(rect=[0, .04, 1, .95])
        png = args.output_dir/f'{key}.png'
        fig.savefig(png)
        plt.close(fig)
        measured['figure'] = str(png.resolve())
        report_cases.append(measured)
        # Self-contained point metadata: no fetch() is needed when opening file://.
        import os
        relative_image = os.path.relpath(image_path, args.output_dir.resolve())
        records = np.column_stack([pixels[in_patch], ranges[in_patch], in_roi[in_patch].astype(float)])
        interactive.append(dict(**measured, image=relative_image,
                                point_pixels_range_inside_roi=np.round(records, 3).tolist(),
                                figure_url=png.name))
        print(json.dumps(measured))
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_bag=manifest['bag'], source_manifest=str((args.samples/'manifest.json').resolve()),
                  lidar_mode=manifest['lidar_mode'], native_slots_per_scan=262144,
                  method='Full raw cloud projected through captured K/D and candidate B; manual depth bands; no road fitting.',
                  limitations=['Image proposals and ROI returns are not verified vehicle membership.',
                               'Expanded search patch is 10 px left/right/top and 100 px below the camera box.',
                               'No calibration adjustment, motion compensation or physical synchronization validation.',
                               'Different scenes do not establish a controlled 10 Hz versus 20 Hz improvement.'],
                  cases=report_cases)
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    payload = json.dumps(interactive).replace('</', '<\\/')
    page = '''<!doctype html><meta charset="utf-8"><title>10 Hz distant vehicle LiDAR inspection</title>
<style>body{margin:24px;background:#101722;color:#eaf0fa;font:16px system-ui}h1{font-size:24px}select,input{margin:10px}canvas{max-width:100%;background:#05080d}a{color:#f8b44c}.note{max-width:1100px;color:#bbc7d8}#count{font-weight:bold;margin:12px 0}button{padding:8px}label{display:inline-block}</style>
<h1>10 Hz distant-vehicle raw LiDAR inspection</h1><p class="note">Orange: raw returns in the selected range band. Yellow: camera vehicle proposal. The search patch extends below the box to expose projection offsets. Counts can include road and background; they are not ground-truth vehicle membership.</p>
<select id="cases"></select><a id="figure" target="_blank">Open detailed raw-point figure</a><br>
<label>Minimum range <input id="lo" type="range" min="0" max="250" step="0.5"><span id="lov"></span></label>
<label>Maximum range <input id="hi" type="range" min="0" max="250" step="0.5"><span id="hiv"></span></label><button id="reset">Reset band</button>
<div id="count"></div><canvas id="view" width="1024" height="768"></canvas>
<p class="note">This is the new 2048×10 recording. Full raw scans were used; no 60 m detection crop, viewer point limit, road geometry, camera 3D model or accumulation was used. Physical timing and the imported camera–LiDAR transform remain provisional.</p>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent),pick=document.getElementById('cases'),lo=document.getElementById('lo'),hi=document.getElementById('hi'),canvas=document.getElementById('view'),ctx=canvas.getContext('2d'),img=new Image();
for(let i=0;i<data.length;i++){let o=document.createElement('option');o.value=i;o.textContent=`Frame ${data[i].sample_id}, ${data[i].elapsed_s.toFixed(1)}s, ${data[i].selected_range_band_m.join('–')} m`;pick.append(o)}
function draw(){if(!img.complete||!img.naturalWidth)return;const row=data[+pick.value],a=+lo.value,b=+hi.value;ctx.drawImage(img,0,0);let n=0,core=0;for(const [u,v,r,inside]of row.point_pixels_range_inside_roi){ctx.fillStyle='#9099a3';ctx.fillRect(u-0.5,v-0.5,1,1);if(r>=a&&r<=b){n++;core+=inside;ctx.fillStyle='#ffaa35';ctx.beginPath();ctx.arc(u,v,2,0,2*Math.PI);ctx.fill()}}const [x,y,x2,y2]=row.camera_roi_xyxy;ctx.strokeStyle='#ffff59';ctx.lineWidth=2;ctx.strokeRect(x,y,x2-x,y2-y);const [l,t,r,d]=row.search_patch_xyxy;ctx.strokeStyle='#bbc7d8';ctx.setLineDash([6,4]);ctx.strokeRect(l,t,r-l,d-t);ctx.setLineDash([]);document.getElementById('count').textContent=`${n} raw returns in search patch / ${core} inside vehicle ROI | source header difference ${row.header_delta_ms.toFixed(1)} ms`;document.getElementById('lov').textContent=a+' m';document.getElementById('hiv').textContent=b+' m'}
function choose(){const row=data[+pick.value];lo.value=row.selected_range_band_m[0];hi.value=row.selected_range_band_m[1];document.getElementById('figure').href=row.figure_url;img.onload=draw;img.src=row.image}pick.onchange=choose;lo.oninput=draw;hi.oninput=draw;document.getElementById('reset').onclick=choose;choose();
</script>'''.replace('PAYLOAD', payload)
    (args.output_dir/'index.html').write_text(page)
    print('Report:', args.output_dir/'report.json')
    print('Viewer:', args.output_dir/'index.html')


if __name__ == '__main__':
    main()
