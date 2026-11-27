#!/usr/bin/python3
"""Render a saved local XYZ/intensity PCD and sensor trajectory as a flat preview."""
import argparse
from pathlib import Path

import numpy as np


def read_map(path):
    with Path(path).open('rb') as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('PCD header has no DATA record')
            words = line.decode('ascii').strip().split()
            if not words or words[0].startswith('#'):
                continue
            header[words[0]] = words[1:]
            if words[0] == 'DATA':
                break
        if (header.get('FIELDS') != ['x', 'y', 'z', 'intensity'] or
                header.get('DATA') != ['binary'] or
                header.get('SIZE') != ['4']*4 or header.get('TYPE') != ['F']*4):
            raise ValueError('expected binary float32 XYZ/intensity PCD from mapping adapter')
        count = int(header['POINTS'][0])
        payload = stream.read()
        if len(payload) != count*16:
            raise ValueError('PCD payload size differs from its declared point count')
        return np.frombuffer(payload, dtype='<f4').reshape(count, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map_directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    points = read_map(args.map_directory/'pointcloud_map.pcd')
    trajectory = np.genfromtxt(args.map_directory/'trajectory.csv', delimiter=',', names=True)
    trajectory = np.atleast_1d(trajectory)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(10, 8), facecolor='#11151b')
    axis.set_facecolor('#11151b')
    # Keep rendering bounded without changing the saved map itself.
    stride = max(1, int(np.ceil(len(points)/150000)))
    shown = points[::stride]
    z = np.clip(shown[:, 2], -3, 3)
    axis.scatter(shown[:, 0], shown[:, 1], c=z, cmap='cividis', s=.25, alpha=.6,
                 vmin=-3, vmax=3, rasterized=True)
    if len(trajectory):
        axis.plot(trajectory['x_m'], trajectory['y_m'], color='#ff9848', linewidth=2,
                  label='Estimated sensor trajectory')
        axis.scatter(trajectory['x_m'][0], trajectory['y_m'][0], c='#55de9b', s=45, label='Start')
        axis.scatter(trajectory['x_m'][-1], trajectory['y_m'][-1], c='#ff9848', s=45, marker='x', label='End')
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('Local map X (m)', color='white')
    axis.set_ylabel('Local map Y (m)', color='white')
    axis.tick_params(colors='white')
    axis.grid(alpha=.15)
    axis.set_title(f'FAST-LIO2 local point-cloud map: {len(points):,} voxels\n'
                   'Local coordinates; no GPS alignment or loop closure', color='white')
    axis.legend(facecolor='#212732', labelcolor='white', loc='upper right')
    output = args.output or args.map_directory/'map_preview.png'
    fig.tight_layout()
    fig.savefig(output, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output.resolve())


if __name__ == '__main__':
    main()
