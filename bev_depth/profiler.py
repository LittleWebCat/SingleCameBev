from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append((time.perf_counter() - start) * 1000.0)

    def summary(self) -> str:
        parts = []
        total = 0.0
        for name, values in self.samples.items():
            if not values:
                continue
            mean = sum(values) / len(values)
            total += mean
            parts.append(f"{name}={mean:.2f}ms")
        fps = 1000.0 / total if total > 0.0 else 0.0
        parts.append(f"approx_fps={fps:.1f}")
        return " ".join(parts)
