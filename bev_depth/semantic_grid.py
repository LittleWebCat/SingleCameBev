from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import BevConfig
from .types import ProjectedDetection


SEMANTIC_CLASS_IDS: dict[str, int] = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "bus": 5,
    "truck": 6,
    "stop sign": 7,
    "traffic light": 8,
    "red": 9,
    "yellow": 10,
    "green": 11,
}
EMPTY_CLASS_ID = 0
UNKNOWN_CLASS_ID = 255


def semantic_grid_shape(bev: BevConfig, cell_size_m: float = 1.0) -> tuple[int, int]:
    if cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    rows = int(round((bev.forward_range_m + bev.rear_range_m) / cell_size_m))
    columns = int(round(bev.lateral_range_m / cell_size_m))
    if rows <= 0 or columns <= 0:
        raise ValueError("BEV dimensions must produce a non-empty semantic grid")
    return rows, columns


def metric_to_semantic_cell(
    x_m_lateral: float,
    z_m_forward: float,
    bev: BevConfig,
    cell_size_m: float = 1.0,
) -> tuple[int, int] | None:
    half_width = bev.lateral_range_m / 2.0
    if not (-half_width <= x_m_lateral <= half_width):
        return None
    if not (-bev.rear_range_m <= z_m_forward <= bev.forward_range_m):
        return None
    rows, columns = semantic_grid_shape(bev, cell_size_m)
    column = math.floor((x_m_lateral + half_width) / cell_size_m)
    row = math.floor((bev.forward_range_m - z_m_forward) / cell_size_m)
    return min(row, rows - 1), min(column, columns - 1)


def render_semantic_grid(
    bev: BevConfig,
    detections: list[ProjectedDetection],
    cell_size_m: float = 1.0,
) -> np.ndarray:
    """Render class IDs; the highest-confidence object wins a shared cell."""
    grid = np.zeros(semantic_grid_shape(bev, cell_size_m), dtype=np.uint8)
    for item in sorted(
        detections, key=lambda detection: detection.detection.confidence, reverse=True
    ):
        if item.x_m_lateral is None or item.z_m_forward is None:
            continue
        cell = metric_to_semantic_cell(
            item.x_m_lateral, item.z_m_forward, bev, cell_size_m
        )
        if cell is None or grid[cell] != EMPTY_CLASS_ID:
            continue
        grid[cell] = SEMANTIC_CLASS_IDS.get(
            item.detection.class_name, UNKNOWN_CLASS_ID
        )
    return grid


def write_semantic_metadata(
    path: str | Path,
    bev: BevConfig,
    cell_size_m: float = 1.0,
) -> None:
    rows, columns = semantic_grid_shape(bev, cell_size_m)
    id_to_class = {str(EMPTY_CLASS_ID): "empty"}
    id_to_class.update({str(value): key for key, value in SEMANTIC_CLASS_IDS.items()})
    id_to_class[str(UNKNOWN_CLASS_ID)] = "unknown"
    metadata = {
        "dtype": "uint8",
        "shape_rows_columns": [rows, columns],
        "physical_width_height_m": [bev.lateral_range_m, bev.forward_range_m + bev.rear_range_m],
        "cell_size_m": cell_size_m,
        "row_0": f"forward +{bev.forward_range_m:g}m",
        "last_row": f"rear -{bev.rear_range_m:g}m",
        "column_0": f"left -{bev.lateral_range_m / 2.0:g}m",
        "last_column": f"right +{bev.lateral_range_m / 2.0:g}m",
        "class_ids": id_to_class,
        "cell_conflict": "highest-confidence detection wins",
    }
    Path(path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
