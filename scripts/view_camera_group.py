"""Show a saved camera group: python3 scripts/view_camera_group.py [config.yaml]."""

import argparse
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'configs/front_camera_group.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    cameras = config['cameras']
    if not cameras:
        parser.error('Camera group is empty')
    settings = config['input']
    latest = {}
    stop = threading.Event()

    def capture(index, camera):
        while not stop.is_set():
            cap = cv2.VideoCapture(camera['source'], cv2.CAP_V4L2)
            try:
                if not cap.isOpened():
                    stop.wait(2)
                    continue
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings['fourcc']))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
                cap.set(cv2.CAP_PROP_FPS, settings['fps'])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                announced = False
                while not stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    latest[index] = (frame, time.monotonic())
                    if not announced:
                        print(f"{camera['name']} | USB {camera['usb_port']}: receiving frames", flush=True)
                        announced = True
            finally:
                cap.release()
            stop.wait(1)

    threads = [threading.Thread(target=capture, args=(i, camera), daemon=True)
               for i, camera in enumerate(cameras)]
    for thread in threads:
        thread.start()
    cols = min(3, len(cameras))
    rows = (len(cameras) + cols - 1) // cols
    window = config.get('name', 'Camera group') + ' - Live (Q / Esc to close)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, cols * 480, rows * 360)
    try:
        while True:
            canvas = np.zeros((rows * 360, cols * 480, 3), dtype=np.uint8)
            now = time.monotonic()
            for i, camera in enumerate(cameras):
                x, y = (i % cols) * 480, (i // cols) * 360
                item = latest.get(i)
                if item:
                    canvas[y:y+360, x:x+480] = cv2.resize(item[0], (480, 360))
                cv2.rectangle(canvas, (x, y), (x+480, y+30), (0, 0, 0), -1)
                label = f"{camera['name']} | USB {camera['usb_port']}"
                cv2.putText(canvas, label, (x+8, y+21), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1)
                if item is None or now - item[1] > 1:
                    cv2.putText(canvas, 'WAITING' if item is None else 'STALE FEED',
                                (x+100, y+190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow(window, canvas)
            if cv2.waitKey(15) & 0xff in (27, ord('q')):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=1)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
