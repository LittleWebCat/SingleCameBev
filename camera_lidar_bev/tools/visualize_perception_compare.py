#!/usr/bin/env python3
"""Draw existing detector outputs; never rerun inference or validate accuracy."""
import argparse
import base64
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CYAN = (36, 223, 242)
ORANGE = (255, 166, 43)
INK = (229, 235, 245)
BACKGROUND = (15, 23, 36)
MUTED = (158, 172, 194)


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def compatible(lidar, camera):
    return camera == 'person' if lidar == 'person' else camera in ('car', 'truck', 'bus', 'motorcycle')


def candidates(results):
    rows = []
    for sample in results['samples']:
        for association in sample['associations']:
            projection = association['projection']
            if not projection['visible']:
                continue
            index = association['lidar_index']
            possible = []
            for camera_index, camera in enumerate(sample['camera_2d_boxes']):
                if not compatible(sample['lidar_boxes'][index]['class'], camera['class']):
                    continue
                box = camera['native_xyxy_px']
                center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
                possible.append({'sample_id': sample['sample_id'], 'lidar_index': index,
                                 'camera_index': camera_index, 'iou': overlap(projection['native_xyxy_px'], box),
                                 'center_offset_px': math.dist(projection['native_center_px'], center),
                                 'camera_center_px': center, 'lidar_center_px': projection['native_center_px'],
                                 'accepted': association['camera_index'] == camera_index})
            if possible:
                rows.append(min(possible, key=lambda c: (-c['iou'], c['center_offset_px'])))
    return rows


def corners(center, sizes, yaw, axis):
    if axis == 'z':
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                             [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    else:
        rotation = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0],
                             [-math.sin(yaw), 0, math.cos(yaw)]])
    points = np.array([[(1 if i & bit else -1)*size/2 for bit, size in zip((1, 2, 4), sizes)] for i in range(8)])
    return points@rotation.T+np.asarray(center)


def project(points, intrinsic, distortion):
    if (points[:, 2] <= .3).any():
        return None
    projected = cv2.projectPoints(points, np.zeros(3), np.zeros(3), intrinsic, distortion)[0].reshape(-1, 2)
    return projected if np.isfinite(projected).all() else None


def wire(draw, points, color, width=2):
    if points is None:
        return
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if j > i:
                draw.line([tuple(points[i]), tuple(points[j])], fill=color, width=width)


def annotate_sample(sample, image, intrinsic, distortion, transform, selected=None, smoke=False):
    canvas = image.convert('RGB').copy()
    draw = ImageDraw.Draw(canvas)
    for i, box in enumerate(sample['camera_2d_boxes']):
        is_selected = selected is not None and i == selected['camera_index']
        color = CYAN if is_selected else (139, 160, 176)
        roi = box['native_xyxy_px']
        draw.rectangle(roi, outline=color, width=4 if is_selected else 1)
        if is_selected:
            draw.text((max(0, roi[0]), max(0, roi[1]-22)), f'C{i}  camera {box["class"]}', font=font(16), fill=color,
                      stroke_width=2, stroke_fill=BACKGROUND)
    for association in sample['associations']:
        if not association['projection']['visible']:
            continue
        i = association['lidar_index']
        box = sample['lidar_boxes'][i]
        points = corners(box['os_sensor_xyz_m'], box['size_lwh_m'], box['yaw_rad'], 'z')
        camera_points = points@transform[:3, :3].T+transform[:3, 3]
        pixels = project(camera_points, intrinsic, distortion)
        is_selected = selected is not None and i == selected['lidar_index']
        color = ORANGE if is_selected or smoke else (142, 103, 65)
        wire(draw, pixels, color, 3 if is_selected else 1)
        if is_selected:
            roi = association['projection']['native_xyxy_px']
            draw.rectangle(roi, outline=color, width=3)
            draw.text((max(0, roi[0]), min(image.height-20, roi[3]+5)), f'L{i}  LiDAR projection', font=font(16), fill=color,
                      stroke_width=2, stroke_fill=BACKGROUND)
    if selected:
        a, b = selected['camera_center_px'], selected['lidar_center_px']
        draw.line([tuple(a), tuple(b)], fill=(248, 249, 251), width=2)
        for p, color in ((a, CYAN), (b, ORANGE)):
            draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill=color, outline=BACKGROUND, width=1)
    if smoke:
        for i, box in enumerate(sample['camera_3d_boxes']):
            points = corners(box['camera_xyz_m'], box['size_lhw_m'], box['camera_rotation_y_rad'], 'y')
            pixels = project(points, intrinsic, distortion)
            color = CYAN if box['class'] == 'car' else (247, 81, 210)
            wire(draw, pixels, color, 3)
            if pixels is not None:
                u, v = pixels.min(axis=0)
                draw.text((max(0, u), max(0, v-22)), f'M{i} SMOKE {box["class"]} {box["score"]:.2f}',
                          font=font(15), fill=color, stroke_width=2, stroke_fill=BACKGROUND)
    return canvas


def case_figure(sample, raw, overlay, selected, title, filename, start_ns):
    canvas = Image.new('RGB', (1720, 1020), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    elapsed = (sample['image_stamp_ns']-start_ns)/1e9
    draw.text((24, 18), title, fill=INK, font=font(29))
    draw.text((24, 62), f'Sample {sample["sample_id"]} | {elapsed:.1f} s after first sampled frame | Cyan: camera 2D | Orange: projected LiDAR 3D',
              fill=MUTED, font=font(19))
    canvas.paste(overlay, (24, 107))
    box = sample['camera_2d_boxes'][selected['camera_index']]['native_xyxy_px']
    association = next(a for a in sample['associations'] if a['lidar_index'] == selected['lidar_index'])
    lidar_roi = association['projection']['native_xyxy_px']
    left, top = max(0, math.floor(min(box[0], lidar_roi[0])-75)), max(0, math.floor(min(box[1], lidar_roi[1])-40))
    right, bottom = min(raw.width, math.ceil(max(box[2], lidar_roi[2])+75)), min(raw.height, math.ceil(max(box[3], lidar_roi[3])+40))
    # Enlarge the selected region while keeping its original aspect ratio.
    scale = min(620/(right-left), 430/(bottom-top))
    crop = overlay.crop((left, top, right, bottom)).resize((round((right-left)*scale), round((bottom-top)*scale)), Image.Resampling.LANCZOS)
    canvas.paste(crop, (1076+(620-crop.width)//2, 107+(430-crop.height)//2))
    camera = sample['camera_2d_boxes'][selected['camera_index']]
    lidar = sample['lidar_boxes'][selected['lidar_index']]
    dx = selected['lidar_center_px'][0]-selected['camera_center_px'][0]
    dy = selected['lidar_center_px'][1]-selected['camera_center_px'][1]
    lines = [f'Overlap (IoU): {100*selected["iou"]:.1f}%',
             f'Center separation: {selected["center_offset_px"]:.1f} pixels',
             f'Horizontal: {dx:+.1f} px | Vertical: {dy:+.1f} px',
             f'Association: {"ACCEPTED" if selected["accepted"] else "NO MATCH"}',
             f'Camera confidence: {camera["score"]:.2f}', f'LiDAR confidence: {lidar["score"]:.2f}',
             f'LiDAR center: {lidar["os_sensor_xyz_m"][0]:.1f} m forward',
             'LiDAR L x W x H: '+ ' x '.join(f'{v:.2f}' for v in lidar['size_lwh_m'])+' m',
             f'Header difference: {sample["pair_delta_ms"]:.1f} ms',
             f'Camera-only SMOKE boxes in this frame: {len(sample["camera_3d_boxes"])}']
    for i, line in enumerate(lines):
        draw.text((1078, 566+i*31), line, fill=INK if i < 4 else MUTED, font=font(20 if i < 4 else 18))
    if selected['accepted']:
        note = 'Highest overlap among the saved candidate pairs. The projected box still sits lower than the camera box.'
    elif selected['iou'] == 0:
        note = 'No camera box overlaps this LiDAR projection. The cyan box is the nearest compatible candidate, not a confirmed counterpart.'
    else:
        note = 'The rectangles overlap slightly, but the association was rejected. The cyan candidate is not a confirmed counterpart.'
    for i, line in enumerate(textwrap.wrap(note, width=115)):
        draw.text((24, 899+i*30), line, fill=INK, font=font(19))
    draw.text((24, 974), 'Existing predictions only. No detector rerun, calibration changes, or accuracy validation.', fill=MUTED, font=font(17))
    canvas.save(filename)


def model_figure(sample, image, overlay, transform, filename):
    canvas = Image.new('RGB', (1720, 1020), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), 'Camera-only 3D and LiDAR 3D: existing predictions', fill=INK, font=font(29))
    draw.text((24, 62), f'Sample {sample["sample_id"]} | Cyan: SMOKE cars | Magenta: SMOKE person | Orange: LiDAR projections', fill=MUTED, font=font(19))
    canvas.paste(overlay, (24, 107))
    left, top, width, height = 1110, 128, 520, 665
    x_limit, lateral_limit = 60, 20
    def point(x, y):
        return left+width*(.5-y/(2*lateral_limit)), top+height*(1-x/x_limit)
    for x in range(0, 61, 10):
        y = point(x, 0)[1]
        draw.line([(left,y),(left+width,y)], fill=(54,65,83))
        draw.text((left-35,y-9),str(x),fill=MUTED,font=font(16))
    for y in range(-20, 21, 10):
        x = point(0,y)[0]
        draw.line([(x,top),(x,top+height)],fill=(54,65,83))
    draw.text((left+140,top-28),'BEV: forward in meters',fill=INK,font=font(18))
    inverse = np.linalg.inv(transform)
    for i, box in enumerate(sample['lidar_boxes']):
        points = corners(box['os_sensor_xyz_m'],box['size_lwh_m'],box['yaw_rad'],'z')
        hull=cv2.convexHull(points[:,:2].astype(np.float32)).reshape(-1,2)
        outline=[point(float(p[0]),float(p[1])) for p in hull]
        draw.line(outline+[outline[0]],fill=ORANGE,width=2)
        center=point(*box['os_sensor_xyz_m'][:2]);draw.text(center,f'L{i}',fill=ORANGE,font=font(15))
    for i, box in enumerate(sample['camera_3d_boxes']):
        camera_points=corners(box['camera_xyz_m'],box['size_lhw_m'],box['camera_rotation_y_rad'],'y')
        points=camera_points@inverse[:3,:3].T+inverse[:3,3]
        hull=cv2.convexHull(points[:,:2].astype(np.float32)).reshape(-1,2)
        outline=[point(float(p[0]),float(p[1])) for p in hull]
        color=CYAN if box['class']=='car' else (247,81,210)
        draw.line(outline+[outline[0]],fill=color,width=3)
        center=np.asarray(box['camera_xyz_m'])@inverse[:3,:3].T+inverse[:3,3]
        draw.text(point(*center[:2]),f'M{i}',fill=color,font=font(18))
    draw.text((24,899),'No object identities have been established between these 3D outputs; the BEV shows separate model predictions.',fill=INK,font=font(19))
    draw.text((24,935),'The person label is a SMOKE prediction (confidence 0.254); LiDAR produced no person box in this sample.',fill=MUTED,font=font(18))
    draw.text((24,974),'Projection uses the saved provisional calibration. No accuracy validation or model reruns.',fill=MUTED,font=font(17))
    canvas.save(filename)


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=root/'evaluation/perception_speed_compare_001.json')
    parser.add_argument('--manifest',type=Path,default=root/'data/perception_compare_001/manifest.json')
    parser.add_argument('--output-dir',type=Path,default=root/'data/perception_visual_analysis_001')
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Choose a new output directory')
    results=json.loads(args.results.read_text());manifest=json.loads(args.manifest.read_text())
    ranked=candidates(results)
    best=max(ranked,key=lambda c:c['iou'])
    rejected=[c for c in ranked if not c['accepted']]
    worst=min(rejected,key=lambda c:(c['iou'],-c['center_offset_px']))
    nonzero=min((c for c in rejected if c['iou']>0),key=lambda c:c['iou'])
    rows={s['sample_id']:s for s in results['samples']};sources={s['sample_id']:s for s in manifest['samples']}
    intrinsic=np.asarray(manifest['calibration']['K']).reshape(3,3)
    distortion=np.asarray(manifest['calibration']['D']);transform=np.asarray(manifest['calibration']['T']).reshape(4,4)
    start_ns=min(s['image_stamp_ns'] for s in manifest['samples'])
    args.output_dir.mkdir(parents=True)
    images={}
    for name,chosen,title in [('minimum_difference',best,'Closest image-box agreement (highest IoU)'),
                              ('maximum_mismatch',worst,'Largest unmatched projected-box separation'),
                              ('worst_overlapping_candidate',nonzero,'Worst candidate with some image overlap')]:
        sample=rows[chosen['sample_id']];raw=Image.open(sources[sample['sample_id']]['image']).convert('RGB')
        overlay=annotate_sample(sample,raw,intrinsic,distortion,transform,chosen)
        filename=args.output_dir/f'{name}.png';case_figure(sample,raw,overlay,chosen,title,filename,start_ns)
        images[name]=filename
    populated=max(results['samples'],key=lambda s:len(s['camera_3d_boxes']))
    raw=Image.open(sources[populated['sample_id']]['image']).convert('RGB')
    overlay=annotate_sample(populated,raw,intrinsic,distortion,transform,smoke=True)
    images['camera_3d_vs_lidar']=args.output_dir/'camera_3d_vs_lidar.png'
    model_figure(populated,raw,overlay,transform,images['camera_3d_vs_lidar'])
    analysis={'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_results':str(args.results),
              'selection_metric':'Best = highest candidate IoU; worst = lowest best-compatible IoU, tie broken by largest nearest-candidate center separation.',
              'minimum_difference':best,'maximum_mismatch':worst,'worst_overlapping_candidate':nonzero,
              'camera_3d_populated_sample':populated['sample_id'],'ground_truth_correspondence_established':False,
              'inference_rerun':False,'accuracy_validation_performed':False,
              'artifacts':{k:str(v.resolve()) for k,v in images.items()}}
    (args.output_dir/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    titles={'minimum_difference':'Closest match','maximum_mismatch':'Largest mismatch',
            'worst_overlapping_candidate':'Worst overlapping candidate','camera_3d_vs_lidar':'Camera 3D vs LiDAR'}
    cards=''.join(f'<section id="{key}" class="card"><h2>{html.escape(titles[key])}</h2><img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{html.escape(titles[key])}"></section>' for key,path in images.items())
    buttons=''.join(f'<button onclick="show(\'{key}\')">{html.escape(title)}</button>' for key,title in titles.items())
    page='''<!doctype html><html><head><meta charset="utf-8"><title>Camera / LiDAR visual comparison</title>
<style>body{margin:0;background:#0f1724;color:#e5ebf5;font:16px system-ui}header{padding:22px}h1{margin:0 0 8px;font-size:26px}p{color:#9eacc2;max-width:1050px;line-height:1.5}nav{display:flex;gap:10px;flex-wrap:wrap}button{background:#223148;border:1px solid #3d5273;color:#e5ebf5;padding:10px 16px;border-radius:8px;cursor:pointer}.card{padding:0 22px 22px}.card img{width:100%;height:auto;border:1px solid #33435b;border-radius:10px}h2{font-size:19px}</style></head><body><header>
<h1>Visual comparison from the saved predictions</h1><p>Closest means highest camera / projected-LiDAR image-box overlap. Largest mismatch means the greatest nearest-candidate separation among zero-overlap projections. These are association diagnostics, not confirmed same-object position errors. The selected extremes are two different LiDAR boxes in sample 007.</p>
<nav>'''+buttons+'''</nav></header>'''+cards+'''
<script>function show(id){document.querySelectorAll('.card').forEach(e=>e.style.display=e.id===id?'block':'none')}show('minimum_difference');</script></body></html>'''
    (args.output_dir/'index.html').write_text(page)
    print(json.dumps({'output_dir':str(args.output_dir),'best':best,'worst':worst,'html':str(args.output_dir/'index.html')},indent=2))


if __name__=='__main__':main()
