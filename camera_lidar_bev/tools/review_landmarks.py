#!/usr/bin/env python3
"""Review mapped stop signs / traffic lights on a top-down LiDAR map and choose which to keep.

Serves a local page (127.0.0.1 only) with the saved map flattened to a top-down density image,
the mapped route, and every landmark from MAP/landmarks/landmarks.json (estimated stop lines
included). Click a landmark or tick it in the list to keep/drop it; Save writes
MAP/landmarks/landmarks_reviewed.json (the original file is never changed).

  /usr/bin/python3 tools/review_landmarks.py [--map maps/local_gps_recording_002] [--port 8090]
"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_pcd_xyz(path):
    with open(path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('ascii', 'replace').strip()
            key, _, value = line.partition(' ')
            header[key] = value
            if key == 'DATA':
                break
        if header['DATA'] != 'binary':
            raise SystemExit(f'{path}: only binary PCD is supported')
        fields = header['FIELDS'].split()
        sizes = [int(s) for s in header['SIZE'].split()]
        if set(sizes) != {4} or any(t != 'F' for t in header['TYPE'].split()):
            raise SystemExit(f'{path}: expected float32 fields')
        data = np.frombuffer(f.read(), np.float32).reshape(int(header['POINTS']), len(fields))
    return data[:, [fields.index('x'), fields.index('y'), fields.index('z')]]


def build_data(map_dir, cell_m=0.5):
    xyz = read_pcd_xyz(map_dir / 'gps_alignment/pointcloud_map_enu.pcd')
    route = np.genfromtxt(map_dir / 'gps_alignment/trajectory_enu.csv', delimiter=',', names=True)
    doc = json.loads((map_dir / 'landmarks/landmarks.json').read_text())
    lm = doc['landmarks']
    # Top-down density of points above the road surface (walls, poles, vegetation, cars).
    ground = np.median(route['z_m']) - 1.7
    keep = (xyz[:, 2] > ground + 0.3) & (xyz[:, 2] < ground + 6.0)
    pts = xyz[keep, :2]
    x0, y0 = np.floor(pts.min(0) / cell_m) * cell_m
    ij = np.floor((pts - [x0, y0]) / cell_m).astype(np.int64)
    w, h = ij.max(0) + 1
    counts = np.bincount(ij[:, 1] * w + ij[:, 0], minlength=w * h)
    occupied = np.nonzero(counts)[0]
    density = np.minimum(255, (np.log1p(counts[occupied]) / np.log1p(np.percentile(counts[occupied], 98)) * 255)).astype(int)
    # Suspect lights: more than 1.5 m below the median of confirmed lights within 30 m.
    lights = [l for l in lm if l['type'] == 'traffic_light' and l['confirmed']]
    for l in lm:
        l['suspect'] = ''
        if l['type'] == 'traffic_light':
            near = [o['position_enu_m'][2] for o in lights if o is not l and
                    np.hypot(o['position_enu_m'][0] - l['position_enu_m'][0], o['position_enu_m'][1] - l['position_enu_m'][1]) < 30]
            if len(near) >= 2 and l['position_enu_m'][2] < np.median(near) - 1.5:
                l['suspect'] = f"{np.median(near) - l['position_enu_m'][2]:.1f} m below nearby lights"
        if not l['confirmed']:
            l['suspect'] = (l['suspect'] + '; ' if l['suspect'] else '') + f"only {l['sightings']} sighting(s)"
        l['keep'] = bool(l['confirmed'] and not l['suspect'])
    step = max(1, len(route) // 1500)
    return {'cell_m': cell_m, 'origin': [float(x0), float(y0)], 'width': int(w), 'height': int(h),
            'cells': occupied.tolist(), 'density': density.tolist(),
            'route': np.stack([route['x_m'][::step], route['y_m'][::step]], 1).round(2).tolist(),
            'landmarks': lm, 'map': map_dir.name, 'stats': doc.get('stats', {})}


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Landmark Review</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--text:#1b1f24;--muted:#5d6672;--line:#d9dee5;--accent:#0b62d6;--keep:#1a7f37;--drop:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#1b2027;--text:#e6e9ee;--muted:#9aa4b1;--line:#2c333d;--accent:#4c9bff;--keep:#4ac26b;--drop:#ff6b60}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;display:flex;height:100vh}
#mapwrap{flex:1;position:relative;min-width:0}canvas{width:100%;height:100%;display:block;cursor:grab;background:#0e1116}
#side{width:430px;max-width:45vw;background:var(--panel);border-left:1px solid var(--line);display:flex;flex-direction:column}
header{padding:14px 16px;border-bottom:1px solid var(--line)}h1{font-size:16px;margin:0 0 4px}
.sub{color:var(--muted);font-size:12px}.bar{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--text);cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}#list{overflow:auto;flex:1}
table{width:100%;border-collapse:collapse;font-size:12.5px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--panel);font-weight:600;color:var(--muted)}tr.sel{outline:2px solid var(--accent);outline-offset:-2px}
tr.dropped td{opacity:.5}.tag{font-size:11px;color:var(--drop)}.legend{position:absolute;left:12px;bottom:12px;background:rgba(14,17,22,.85);color:#e6e9ee;padding:8px 10px;border-radius:6px;font-size:12px;line-height:1.6}
#status{font-size:12px;color:var(--muted);margin-top:8px;min-height:1.2em}
</style></head><body>
<div id="mapwrap"><canvas id="c"></canvas>
<div class="legend">⬢ red: stop sign &nbsp; ● yellow: traffic light &nbsp; ▬ white: estimated stop line<br>
solid = keep, hollow + ✕ = drop · drag to pan, wheel to zoom, click to toggle</div></div>
<div id="side"><header><h1>Landmark review</h1><div class="sub" id="sub"></div>
<div class="bar"><button id="confirmedOnly">Keep confirmed, drop suspects</button><button id="all">Keep all</button>
<button id="none">Drop all</button><button class="primary" id="save">Save</button></div><div id="status"></div></header>
<div id="list"><table><thead><tr><th>Keep</th><th>ID</th><th>Type</th><th>Seen</th><th>Spread</th><th>Notes</th></tr></thead><tbody id="rows"></tbody></table></div></div>
<script>
const D = __DATA__;
const L = D.landmarks, cv = document.getElementById('c'), ctx = cv.getContext('2d');
let sel = null, view = {s: 2, x: 0, y: 0};
const mapImg = document.createElement('canvas'); mapImg.width = D.width; mapImg.height = D.height;
{const g = mapImg.getContext('2d'), im = g.createImageData(D.width, D.height);
 D.cells.forEach((c, k) => {const i = c % D.width, j = Math.floor(c / D.width), p = ((D.height - 1 - j) * D.width + i) * 4, v = 60 + D.density[k] * 0.75;
   im.data[p] = v; im.data[p + 1] = v; im.data[p + 2] = v + 8; im.data[p + 3] = 255;}); g.putImageData(im, 0, 0);}
const toScreen = (x, y) => [(x - D.origin[0]) / D.cell_m * view.s + view.x, (D.height - (y - D.origin[1]) / D.cell_m) * view.s + view.y];
function fit() {const r = cv.getBoundingClientRect(); cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio;
  const xs = L.map(l => l.position_enu_m[0]).concat(D.route.map(p => p[0])), ys = L.map(l => l.position_enu_m[1]).concat(D.route.map(p => p[1]));
  const minx = Math.min(...xs) - 20, maxx = Math.max(...xs) + 20, miny = Math.min(...ys) - 20, maxy = Math.max(...ys) + 20;
  view.s = Math.min(cv.width / ((maxx - minx) / D.cell_m), cv.height / ((maxy - miny) / D.cell_m));
  const [a, b] = toScreen(minx, maxy); view.x -= a; view.y -= b; const [c2, d2] = toScreen(maxx, miny);
  view.x += (cv.width - c2) / 2; view.y += (cv.height - d2) / 2; draw();}
function octagon(x, y, r) {ctx.beginPath(); for (let k = 0; k < 8; k++) {const a = Math.PI / 8 + k * Math.PI / 4; ctx.lineTo(x + r * Math.cos(a), y + r * Math.sin(a));} ctx.closePath();}
function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#0e1116'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(mapImg, view.x, view.y, D.width * view.s, D.height * view.s);
  ctx.strokeStyle = 'rgba(60,220,90,.9)'; ctx.lineWidth = 2 * devicePixelRatio; ctx.beginPath();
  D.route.forEach((p, i) => {const [x, y] = toScreen(p[0], p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);}); ctx.stroke();
  const r = Math.max(5, Math.min(14, view.s * 1.4)) * devicePixelRatio;
  L.forEach(l => {
    const [x, y] = toScreen(l.position_enu_m[0], l.position_enu_m[1]), stop = l.type === 'stop_sign';
    if (stop && l.stop_line_estimated && l.keep) {ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3 * devicePixelRatio; ctx.beginPath();
      l.stop_line_estimated.endpoints_enu_m.forEach((e, i) => {const [a, b] = toScreen(e[0], e[1]); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b);}); ctx.stroke();}
    const col = stop ? '#ff3b30' : '#ffcc00';
    stop ? octagon(x, y, r) : (ctx.beginPath(), ctx.arc(x, y, r * 0.8, 0, 2 * Math.PI));
    if (l.keep) {ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5 * devicePixelRatio; ctx.stroke();}
    else {ctx.strokeStyle = col; ctx.lineWidth = 2 * devicePixelRatio; ctx.stroke(); ctx.beginPath(); ctx.moveTo(x - r * .6, y - r * .6); ctx.lineTo(x + r * .6, y + r * .6);
      ctx.moveTo(x + r * .6, y - r * .6); ctx.lineTo(x - r * .6, y + r * .6); ctx.stroke();}
    if (sel === l.id) {ctx.strokeStyle = '#4c9bff'; ctx.lineWidth = 3 * devicePixelRatio; ctx.beginPath(); ctx.arc(x, y, r * 1.6, 0, 2 * Math.PI); ctx.stroke();}
    ctx.fillStyle = '#fff'; ctx.font = `${11 * devicePixelRatio}px system-ui`; ctx.fillText(l.id, x + r + 2, y - r * .3);
  });
}
function rows() {
  document.getElementById('rows').innerHTML = L.map(l => `<tr data-id="${l.id}" class="${l.keep ? '' : 'dropped'} ${sel === l.id ? 'sel' : ''}">
   <td><input type="checkbox" ${l.keep ? 'checked' : ''} aria-label="keep landmark ${l.id}"></td><td>${l.id}</td>
   <td>${l.type === 'stop_sign' ? 'Stop sign' : 'Traffic light'}</td><td>${l.sightings}</td><td>${l.spread_m} m</td>
   <td>${l.cameras.join(', ')}<br>height ${l.position_enu_m[2]} m${l.suspect ? `<br><span class="tag">${l.suspect}</span>` : ''}</td></tr>`).join('');
  const k = L.filter(l => l.keep);
  document.getElementById('sub').textContent = `${D.map}: keeping ${k.filter(l => l.type === 'stop_sign').length} stop signs and ${k.filter(l => l.type === 'traffic_light').length} traffic lights of ${L.length} candidates`;
}
function refresh() {rows(); draw();}
document.getElementById('rows').addEventListener('click', e => {const tr = e.target.closest('tr'); if (!tr) return; const l = L[+tr.dataset.id];
  if (e.target.type === 'checkbox') l.keep = e.target.checked; sel = l.id; refresh();});
let drag = null, moved = false;
cv.addEventListener('mousedown', e => {drag = [e.clientX, e.clientY]; moved = false; cv.style.cursor = 'grabbing';});
addEventListener('mouseup', e => {cv.style.cursor = 'grab'; if (drag && !moved) {const b = cv.getBoundingClientRect(), px = (e.clientX - b.left) * devicePixelRatio, py = (e.clientY - b.top) * devicePixelRatio;
  let best = null, bd = 18 * devicePixelRatio; L.forEach(l => {const [x, y] = toScreen(...l.position_enu_m); const d = Math.hypot(x - px, y - py); if (d < bd) {bd = d; best = l;}});
  if (best) {best.keep = !best.keep; sel = best.id; refresh(); document.querySelector(`tr[data-id="${best.id}"]`)?.scrollIntoView({block: 'nearest'});}} drag = null;});
addEventListener('mousemove', e => {if (!drag) return; const dx = e.clientX - drag[0], dy = e.clientY - drag[1]; if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
  view.x += dx * devicePixelRatio; view.y += dy * devicePixelRatio; drag = [e.clientX, e.clientY]; draw();});
cv.addEventListener('wheel', e => {e.preventDefault(); const b = cv.getBoundingClientRect(), px = (e.clientX - b.left) * devicePixelRatio, py = (e.clientY - b.top) * devicePixelRatio,
  f = e.deltaY < 0 ? 1.25 : 0.8; view.x = px - (px - view.x) * f; view.y = py - (py - view.y) * f; view.s *= f; draw();}, {passive: false});
const setAll = fn => {L.forEach(l => l.keep = fn(l)); refresh();};
document.getElementById('confirmedOnly').onclick = () => setAll(l => l.confirmed && !l.suspect);
document.getElementById('all').onclick = () => setAll(() => true);
document.getElementById('none').onclick = () => setAll(() => false);
document.getElementById('save').onclick = async () => {const st = document.getElementById('status'); st.textContent = 'Saving…';
  try {const r = await fetch('/save', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(L.map(l => ({id: l.id, keep: l.keep})))});
    const j = await r.json(); st.textContent = r.ok ? `Saved ${j.kept} kept landmarks to ${j.file}` : `Save failed: ${j.error}`;} catch (err) {st.textContent = 'Save failed: ' + err;}};
addEventListener('resize', fit); rows(); fit();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()
    data = build_data(args.map)
    page = PAGE.replace('__DATA__', json.dumps(data)).encode()
    out = args.map / 'landmarks/landmarks_reviewed.json'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, body, kind='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, page, 'text/html; charset=utf-8') if self.path in ('/', '/index.html') else self.reply(404, b'{}')

        def do_POST(self):
            if self.path != '/save':
                return self.reply(404, b'{}')
            try:
                choices = {int(c['id']): bool(c['keep']) for c in json.loads(self.rfile.read(int(self.headers['Content-Length'])))}
                source = json.loads((args.map / 'landmarks/landmarks.json').read_text())
                for l in source['landmarks']:
                    l['keep'] = choices.get(l['id'], False)
                source['reviewed_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
                source['review'] = 'keep flags chosen in tools/review_landmarks.py; only kept landmarks should be used'
                out.write_text(json.dumps(source, indent=2) + '\n')
                kept = sum(l['keep'] for l in source['landmarks'])
                print(f'Saved {kept} kept of {len(source["landmarks"])} to {out}', flush=True)
                self.reply(200, json.dumps({'kept': kept, 'file': str(out.relative_to(ROOT))}).encode())
            except Exception as e:  # report to the page
                self.reply(400, json.dumps({'error': str(e)}).encode())

    print(f'Review page: http://127.0.0.1:{args.port}/  ({len(data["landmarks"])} landmarks, '
          f'{len(data["cells"])} map cells); saves to {out}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
