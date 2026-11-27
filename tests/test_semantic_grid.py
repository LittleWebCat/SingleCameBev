import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bev_depth.config import BevConfig
from bev_depth.semantic_grid import (
    metric_to_semantic_cell,
    render_semantic_grid,
    write_semantic_metadata,
)
from bev_depth.types import Detection, ProjectedDetection


def projected(class_name, confidence, x_m, z_m):
    return ProjectedDetection(
        Detection(class_name, confidence, (0.0, 0.0, 10.0, 10.0)),
        10.0,
        x_m,
        z_m,
        (0, 0),
        "depth_map",
        "middle",
    )


class SemanticGridTest(unittest.TestCase):
    def setUp(self):
        self.bev = BevConfig(
            forward_range_m=80.0,
            lateral_range_m=80.0,
            pixels_per_meter=5,
            rear_range_m=40.0,
        )

    def test_grid_is_120_rows_by_80_columns_with_class_ids(self):
        grid = render_semantic_grid(
            self.bev,
            [
                projected("car", 0.9, 0.0, 0.0),
                projected("green", 0.8, -10.0, 20.0),
            ],
        )

        self.assertEqual(grid.shape, (120, 80))
        self.assertEqual(grid.dtype, np.uint8)
        self.assertEqual(grid[80, 40], 3)
        self.assertEqual(grid[60, 30], 11)
        self.assertEqual(grid[0, 0], 0)

    def test_highest_confidence_wins_when_objects_share_a_cell(self):
        grid = render_semantic_grid(
            self.bev,
            [
                projected("person", 0.6, 1.2, 10.2),
                projected("truck", 0.9, 1.8, 10.8),
            ],
        )

        self.assertEqual(grid[69, 41], 6)

    def test_edges_map_inside_matrix_and_outside_is_rejected(self):
        self.assertEqual(metric_to_semantic_cell(-40.0, 80.0, self.bev), (0, 0))
        self.assertEqual(metric_to_semantic_cell(40.0, -40.0, self.bev), (119, 79))
        self.assertIsNone(metric_to_semantic_cell(40.1, 0.0, self.bev))

    def test_metadata_describes_axes_and_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            write_semantic_metadata(path, self.bev)
            metadata = json.loads(path.read_text())

        self.assertEqual(metadata["shape_rows_columns"], [120, 80])
        self.assertEqual(metadata["class_ids"]["0"], "empty")
        self.assertEqual(metadata["class_ids"]["3"], "car")
        self.assertEqual(metadata["class_ids"]["11"], "green")


if __name__ == "__main__":
    unittest.main()
